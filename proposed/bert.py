import time

import torch
import wordsegment as ws
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertForSequenceClassification

from .abstractfactory import AbstractContainer, AbstractLoader, AbstractDataset
from .constants import *
from .handlers import TrainHandler, PredictHandler


class BertContainer(AbstractContainer):
    def __init__(self, model_name=None):
        super().__init__()
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = "bert_" + str(int(time.time()))

    @staticmethod
    def loader(split_ratio, batch_size, segment=False):
        return BertLoader(split_ratio, batch_size, segment)

    def train(self, batch_size=32, split_ratio=0.99, n_epochs=4, alpha=2e-5, segment=False, merge=None,
              save_logits=False):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=False,
                                                              output_hidden_states=False, cache_dir=CACHE_PATH)
        loader = self.loader(split_ratio, batch_size)
        self.do_train(loader, n_epochs, alpha, model, save_logits)

    def predict(self, model_name, checkpoint_name, batch_size=32, segment=False, merge=None):
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, output_attentions=False,
                                                              output_hidden_states=False, cache_dir=CACHE_PATH)
        loader = self.loader(None, batch_size)
        self.do_predict(checkpoint_name, loader, model, model_name)

    def do_train(self, loader, n_epochs, alpha, model, save_logits):
        train_loader, val_loader = loader.training_loaders()
        BertTrainHandler(model, train_loader, val_loader, alpha, n_epochs, self.device, self.model_name, save_logits).train()
        self.do_predict("", loader, model, self.model_name)

    def do_predict(self, checkpoint_name, loader, model, model_name):
        test_loader = loader.testing_loaders()
        BertPredictHandler(self.device, model, test_loader, model_name, checkpoint_name).predict()


class BertLoader(AbstractLoader):
    def __init__(self, split_ratio, b_size, segment=False):
        super().__init__(split_ratio, b_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = AbstractDataset(mode='train')
        self.test_dataset = AbstractDataset(mode='test')
        self.is_segment = segment
        if self.is_segment:
            ws.load()

    def segment(self, sentence):
        def custom_word(word):
            if len(word) > CHAR_LIMIT and word[0] == HASH:
                new_word_segments = [HASH + t for t in ws.segment(word)]
                return ' '.join(new_word_segments)
            return word

        if self.is_segment:
            words = [custom_word(x) for x in sentence.strip.split()]
            return ' '.join(words)
        return sentence

    def collate(self, samples):
        labels = [label for sent, label in samples]
        encoded_dict = self.tokenizer.batch_encode_plus(
            [self.segment(sent) for sent, label in samples],
            add_special_tokens=True, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt',)
        return encoded_dict['input_ids'], encoded_dict['attention_mask'], torch.LongTensor(labels), encoded_dict


class BertTrainHandler(TrainHandler):
    def __init__(self, model, train_loader, val_loader, alpha, n_epochs, device, model_name, save_logits):
        super().__init__(model, train_loader, val_loader, alpha, n_epochs, device, model_name, save_logits)
        self.optimizer = AdamW(model.parameters(), lr=self.alpha, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0, len(train_loader) * n_epochs)

    def create_loss_logits(self, batch):
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        self.model.zero_grad()
        loss, logits = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        return loss, logits, b_labels

    def validate(self, batch):
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)
        with torch.no_grad():
            (loss, logits) = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        return loss, logits, b_labels


class BertPredictHandler(PredictHandler):
    def create_logits(self, batch):
        b_input_ids = batch[0].to(self.device)
        b_input_mask = batch[1].to(self.device)
        b_labels = batch[2].to(self.device)

        b_input_ids = torch.squeeze(b_input_ids)
        b_input_mask = torch.squeeze(b_input_mask)

        with torch.no_grad():
            (_, outputs) = self.model(b_input_ids,
                                      token_type_ids=None,
                                      attention_mask=b_input_mask,
                                      labels=b_labels)
        return outputs
