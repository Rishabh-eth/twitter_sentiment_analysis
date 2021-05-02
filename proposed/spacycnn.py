#!/usr/bin/env python
"""Train a convolutional neural network text classifier on the
Twitter dataset, using the TextCategorizer component. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import os
import time
import random
from pathlib import Path

import spacy
from .constants import *
from spacy.util import minibatch, compounding


def get_data(split_ratio):
    f = open(TRAIN_POS_PATH, 'r', encoding='utf-8')
    positive = f.read().split('\n')
    f.close()
    f = open(TRAIN_NEG_PATH, 'r', encoding='utf-8')
    negative = f.read().split('\n')
    f.close()
    pos_label = [{'POSITIVE': True, 'NEGATIVE': False}] * len(positive)
    neg_label = [{'POSITIVE': False, 'NEGATIVE': True}] * len(negative)
    labels = pos_label + neg_label
    texts = positive + negative
    data = list(zip(texts, labels))
    random.shuffle(data)
    n_dev = int((1-split_ratio) * len(data))
    train_data, dev_data = data[n_dev:], data[:n_dev]
    return tuple(zip(*train_data)), tuple(zip(*dev_data))


class SpacyContainer(object):
    def __init__(self, model_name):
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = "bert_" + str(int(time.time()))

    def train(self, split_ratio=0.90, model=None, n_epochs=20, init_tok2vec=None):
        output_dir = CHECKPOINT_PATH + self.model_name
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()

        if model is not None:
            nlp = spacy.load(model)  # load existing spaCy model
            print("Loaded model '%s'" % model)
        else:
            nlp = spacy.blank("en")  # create blank Language class
            print("Created blank 'en' model")

        # add the text classifier to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "textcat" not in nlp.pipe_names:
            textcat = nlp.create_pipe(
                "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
            )
            nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
        else:
            textcat = nlp.get_pipe("textcat")

        # add label to text classifier
        textcat.add_label("POSITIVE")
        textcat.add_label("NEGATIVE")

        # load the IMDB dataset
        print("Loading IMDB data...")
        (train_texts, train_cats), (dev_texts, dev_cats) = get_data(split_ratio)
        print(
            "Using ({} training, {} evaluation) samples".format(
                len(train_texts), len(dev_texts)
            )
        )
        train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

        # get names of other pipes to disable them during training
        pipe_exceptions = ["textcat", "trfs_wordpiecer", "trf_tok2vec"]
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
        with nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = nlp.begin_training(n_process=10)
            if init_tok2vec is not None:
                with init_tok2vec.open("rb") as file_:
                    textcat.model.tok2vec.from_bytes(file_.read())
            print("Training the model...")
            print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
            batch_sizes = compounding(4.0, 64.0, 1.001)
            for i in range(n_epochs):
                losses = {}
                # batch up the examples using spaCy's minibatch
                random.shuffle(train_data)
                batches = minibatch(train_data, size=batch_sizes)
                for bb, batch in enumerate(batches):
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
                with textcat.model.use_params(optimizer.averages):
                    # evaluate on the dev data split off in load_data()
                    scores = self.evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
                print(
                    "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                        losses["textcat"],
                        scores["textcat_p"],
                        scores["textcat_r"],
                        scores["textcat_f"],
                    )
                )

                with nlp.use_params(optimizer.averages):
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    path = '{}/{}'.format(output_dir, i)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    nlp.to_disk(path)
        print("Saved models to", output_dir)

    def evaluate(self, tokenizer, textcat, texts, cats):
        docs = (tokenizer(text) for text in texts)
        tp = 0.0  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 0.0  # True negatives
        for i, doc in enumerate(textcat.pipe(docs)):
            gold = cats[i]
            for label, score in doc.cats.items():
                if label not in gold:
                    continue
                if label == "NEGATIVE":
                    continue
                if score >= 0.5 and gold[label] >= 0.5:
                    tp += 1.0
                elif score >= 0.5 and gold[label] < 0.5:
                    fp += 1.0
                elif score < 0.5 and gold[label] < 0.5:
                    tn += 1
                elif score < 0.5 and gold[label] >= 0.5:
                    fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if (precision + recall) == 0:
            f_score = 0.0
        else:
            f_score = 2 * (precision * recall) / (precision + recall)
        return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}

    def predict(self, model_name, checkpoint_name):
        output_dir = os.path.join(CHECKPOINT_PATH, model_name, checkpoint_name)
        f = open(TEST_PATH, 'r')
        data = f.read().split('\n')
        f.close()

        nlp2 = spacy.load(output_dir)
        submission_filepath = os.path.join(CHECKPOINT_PATH, model_name, model_name + ".csv")
        f = open(submission_filepath, 'w')
        f.write("Id,Prediction\n")
        for line in data[:-1]:
            words = line.strip().split(',')
            index = int(words[0])
            msg = ','.join(words[1:])
            doc2 = nlp2(msg)
            cats = doc2.cats
            if cats['POSITIVE'] > cats['NEGATIVE']:
                f.write("%d,%d\n" % (index, 1))
            else:
                f.write("%d,%d\n" % (index, -1))
        f.close()
        print('Submission File available at: {}'.format(submission_filepath))

