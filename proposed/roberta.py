import time

from transformers import RobertaForSequenceClassification, RobertaTokenizer

from .bert import BertContainer
from .constants import *


class RobertaContainer(BertContainer):
    def __init__(self, model_name=None):
        super().__init__()
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = "roberta_" + str(int(time.time()))

    @staticmethod
    def loader(train_fraction, batch_size, segment=False, merge=None):
        _loader = BertContainer.loader(train_fraction, batch_size, segment)
        if merge is not None:
            _loader.tokenizer = RobertaTokenizer.from_pretrained('roberta-base', merges_file=merge)
        else:
            _loader.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        return _loader

    def train(self, batch_size=32, split_ratio=0.99, n_epochs=4, alpha=2e-5, segment=False, merge=None,
              save_logits=False):
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', cache_dir=CACHE_PATH)
        data_loader = self.loader(split_ratio, batch_size, segment, merge)
        self.do_train(data_loader, n_epochs, alpha, model, save_logits)

    def predict(self, model_name, checkpoint_name, batch_size=32, segment=False, merge=None):
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', cache_dir=CACHE_PATH)
        data_loader = self.loader(None, batch_size, segment, merge)
        self.do_predict(checkpoint_name, data_loader, model, model_name)

