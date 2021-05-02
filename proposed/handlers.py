import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .constants import *


def count_correct(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    return np.sum(pred_flat == labels.flatten())


class TrainHandler(object):
    def __init__(self, model, train_loader, val_loader, alpha, n_epochs, device, model_name, save_logits=False,
                 optimizer=None, scheduler=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.device = device
        self.model_name = model_name

        self.save_logits = save_logits
        self.optimizer = optimizer
        self.scheduler = scheduler

    def create_loss_logits(self, batch):
        raise NotImplementedError

    def validate(self, batch):
        raise NotImplementedError

    def train(self):
        for epoch_i in range(0, self.n_epochs):
            print('Starting Epoch.. {} of {}'.format(epoch_i + 1, self.n_epochs))
            self.train_single_epoch()
            self.validate_single_epoch(epoch_i)

    def validate_single_epoch(self, epoch_i):
        self.model.eval()
        cum_loss, correct, total = 0, 0, 0
        true_labels, confidences = list(), list()
        for batch in self.val_loader:
            loss, logits, b_labels = self.validate(batch)
            true_labels.append(b_labels.to("cpu").numpy())
            confidences.append(logits.detach().cpu())

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            correct += count_correct(logits, label_ids)
            total += len(label_ids.flatten())

            cum_loss += loss.mean().item()
        avg_val_loss = cum_loss / len(self.val_loader)
        print("Val Loss: {0:.3f},\tVal Acc.: {0:.3f}".format(avg_val_loss, (correct / total * 100.)))
        true_labels = np.concatenate(true_labels)
        confidences = np.concatenate(confidences)
        self.dump_checkpoint(epoch_i)
        if self.save_logits:
            self.dump_train_logits(confidences, epoch_i, true_labels)

    def train_single_epoch(self):
        self.model.train()
        cum_loss, correct, total = 0, 0, 0
        n_batches = len(self.train_loader)
        for step, batch in tqdm(enumerate(self.train_loader), total=n_batches):
            loss, logits, b_labels = self.create_loss_logits(batch)
            cum_loss += loss.mean().item()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            correct += count_correct(logits, label_ids)
            total += len(label_ids.flatten())
        avg_train_loss = cum_loss / len(self.train_loader)  # I'm using average loss
        print("Training loss: {0:.3f},\tTraining Acc.: {0:.3f}".format(avg_train_loss, (correct / total * 100.)))

    def dump_checkpoint(self, epoch_i):
        path = CHECKPOINT_PATH + self.model_name
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), os.path.join(path, 'weights_{}.pth'.format(epoch_i)))

    def dump_train_logits(self, confidences, epoch_i, true_labels):
        path = CHECKPOINT_PATH + self.model_name
        logits_path = os.path.join(path, str(epoch_i) + ".logits.csv")
        df = pd.DataFrame()
        df['True_Label'] = true_labels
        df['Logits_0'] = [x[0] for x in confidences]
        df['Logits_1'] = [x[1] for x in confidences]
        df.to_csv(logits_path, index=False)


class PredictHandler(object):
    def __init__(self, device, model, test_loader, model_name, checkpoint_name=""):
        self.device = device
        self.model = model.to(device)
        self.test_loader = test_loader
        self.model_name = model_name
        self.checkpoint_name = checkpoint_name
        self.model_path = CHECKPOINT_PATH + model_name + "/"

    def create_logits(self, batch):
        raise NotImplementedError

    def predict(self):
        print('Generating Submission on Test Data...')

        if self.checkpoint_name != "":
            checkpoint_file = self.model_path + self.checkpoint_name
            state_dict = torch.load(checkpoint_file, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        predictions, confidences = list(), list()
        for batch in tqdm(self.test_loader):
            outputs = self.create_logits(batch)
            outputs = outputs.detach().cpu().numpy()
            predicted = np.argmax(outputs, axis=1)
            predicted = 2 * predicted - 1
            predictions.append(predicted)
            confidences.append(outputs)

        final_preds = np.concatenate(predictions, axis=0)
        final_confidences = np.concatenate(confidences)
        self.dump_predictions(final_confidences, final_preds)

    def dump_predictions(self, final_confidences, final_preds):
        submission_filepath = self.model_path + self.model_name + ".csv"
        df = pd.DataFrame()
        df['Id'] = list(range(1, len(final_preds)+1))
        df['Prediction'] = final_preds
        df.to_csv(submission_filepath, index=False)
        print('Submission File available at: {}'.format(submission_filepath))

        logits_filepath = self.model_path + self.model_name + ".logits.csv"
        df = pd.DataFrame()
        df['Id'] = list(range(1, len(final_preds)+1))
        df['Prediction'] = final_preds
        df['Logits_0'] = [x[0] for x in final_confidences]
        df['Logits_1'] = [x[1] for x in final_confidences]
        df.to_csv(logits_filepath, index=False)