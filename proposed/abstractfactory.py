from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch
from .constants import *


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


class AbstractContainer:
    def __init__(self):
        self.device = get_device()

    def train(self, batch_size=32, split_ratio=0.99, n_epochs=4, alpha=2e-5, segment=False, merge=None,
              save_logits=False):
        raise NotImplementedError

    def predict(self, model_name, checkpoint_name, batch_size=32, segment=False, merge=None):
        raise NotImplementedError


class AbstractLoader:
    def __init__(self, split_ratio, b_size):
        self.tokenizer = None
        self.train_dataset = None
        self.test_dataset = None
        self.b_size = b_size
        self.split_ratio = split_ratio

    def collate(self, samples):
        raise NotImplementedError

    def training_loaders(self):
        train_size = int(self.split_ratio * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        train_dataset, val_dataset = random_split(self.train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset,), batch_size=self.b_size,
                                  collate_fn=self.collate)
        val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=self.b_size,
                                collate_fn=self.collate)
        return train_loader, val_loader

    def testing_loaders(self):
        return DataLoader(self.test_dataset, sampler=SequentialSampler(self.test_dataset), batch_size=self.b_size,
                          collate_fn=self.collate)


class AbstractDataset(Dataset):
    def __init__(self, mode='train'):
        if mode == 'train':
            t0, l0 = self.read_file(TRAIN_POS_PATH, 1)
            t1, l1 = self.read_file(TRAIN_NEG_PATH, 0)
            self.texts = t0 + t1
            self.labels = l0 + l1
        else:
            self.texts, self.labels = self.read_testfile(TEST_PATH)

    @staticmethod
    def read_file(path, label):
        texts, labels = list(), list()
        for sent in open(path):
            texts.append(sent)
            labels.append(label)
        return texts, labels

    @staticmethod
    def read_testfile(path):
        texts, labels = list(), list()
        for sent in open(path):
            texts.append(sent.split(',', maxsplit=1)[1])
            labels.append(1)
        return texts, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
