from .roberta import RobertaContainer
from .bert import BertContainer
from .spacycnn import SpacyContainer
from .xlnet import XLNetContainer

'''
    Multiple code snippets of this package can be exact or modified versions of the links mentioned below.
    1. http://mccormickml.com/2019/07/22/BERT-fine-tuning/
    2. https://spacy.io/usage/training
'''


def train_model(model_type, exp_name=None):
    if model_type == "roberta":
        model_container = RobertaContainer(exp_name)
        model_container.train(split_ratio=0.90, n_epochs=2, alpha=2e-5, save_logits=True)
    elif model_type == "bert":
        model_container = BertContainer(exp_name)
        model_container.train(split_ratio=0.90, n_epochs=2, alpha=2e-5, save_logits=True)
    elif model_type == "spacycnn":
        model_container = SpacyContainer(exp_name)
        model_container.train(split_ratio=0.90, n_epochs=20)
    elif model_type == "xlnet":
        model_container = XLNetContainer(exp_name)
        model_container.train(split_ratio=0.90, n_epochs=2, alpha=2e-5, save_logits=True)
    else:
        raise RuntimeError('Unknown Model Type: {}'.format(model_type))


def predict_model(model_type, exp_name, checkpoint_name):
    if model_type == "roberta":
        model_container = RobertaContainer(exp_name)
    elif model_type == "bert":
        model_container = BertContainer(exp_name)
    elif model_type == "spacycnn":
        model_container = SpacyContainer(exp_name)
    elif model_type == "xlnet":
        model_container = XLNetContainer(exp_name)
    else:
        raise RuntimeError('Unknown Model Type: {}'.format(model_type))
    model_container.predict(model_name=exp_name, checkpoint_name=checkpoint_name)
