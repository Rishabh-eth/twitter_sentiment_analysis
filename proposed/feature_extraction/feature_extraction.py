import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import XLMRobertaForSequenceClassification
from transformers import XLMRobertaTokenizer

MODEL_KEY = 'xlm-roberta-base'
MODEL_WEIGHTS_PATH = 'weights_0.pth'
DATA_PATH = 'train_pos.txt'
BATCH_SIZE = 32
MAX_LEN = 140


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("PyTorch Device: {}".format(device))


class CustomRobertaClassificationHead(nn.Module):
    """
        -   This class is modified version of RobertaClassificationHead in `transformers.modeling_roberta` package
        -   Objective is to remove the last dense layer from the architecture while keeping the weight of the layers
            intact.
        -   Since this class does not inherit from RobertaClassificationHead, it needs the original object instance of
            pre-trained component to copy layers including weights

    """
    def __init__(self, orig_classifier):
        super().__init__()
        self.dense = orig_classifier.dense
        self.dropout = orig_classifier.dropout

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return x


def get_feature_extractor_model(model_name, model_weight_path):
    print("Loading Model Architecture...")
    model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    custom_model = XLMRobertaForSequenceClassification.from_pretrained(
        model_name,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    print("Loading Weights in Original Model...")
    model.load_state_dict(torch.load(model_weight_path))
    print("Loading Weights in Custom Model...")
    custom_model.load_state_dict(torch.load(model_weight_path))

    # Set both the models to evaluation mode
    model.eval()
    custom_model.eval()

    # Change the last component (aka self.classifier) in custom model
    custom_model.classifier = CustomRobertaClassificationHead(custom_model.classifier)

    # Make sure that weights are indeed the same in both the model
    assert torch.all(torch.eq(list(model.roberta.children())[2].dense.weight,
                              list(custom_model.roberta.children())[2].dense.weight
                              )).item()
    assert torch.all(torch.eq(list(model.classifier.children())[0].weight,
                              list(custom_model.classifier.children())[0].weight
                              )).item()
    return custom_model


def read_data(data_path):
    X = pd.read_csv(data_path, delimiter='/n', header=None, error_bad_lines=False)
    sentences = []
    for i in range(int(len(X.values))):
        sentences.append(X.values[i, 0])
    return sentences


def tokenize_data(sentences, tokenizer, MAX_LEN=140):
    input_ids = []
    for index, sent in enumerate(sentences):
        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        )
        input_ids.append(encoded_sent)

    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                              value=0, truncating="post", padding="post")
    return input_ids


def get_attention_masks(input_ids):
    attention_masks = []
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


def create_dataloader(input_ids, attention_masks):
    train_inputs = torch.tensor(input_ids)
    train_masks = torch.tensor(attention_masks)
    # Create the DataLoader
    train_data = TensorDataset(train_inputs, train_masks)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    return train_dataloader


def main():
    model = get_feature_extractor_model(MODEL_KEY, MODEL_WEIGHTS_PATH)
    sentences = read_data(data_path=DATA_PATH)
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_KEY)
    input_ids = tokenize_data(sentences, tokenizer=tokenizer, MAX_LEN=MAX_LEN)
    attention_masks = get_attention_masks(input_ids)
    dataloader = create_dataloader(input_ids, attention_masks)

    outputs = None
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch
        with torch.no_grad():
            b_output = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
            if outputs is None:
                outputs = b_output[0]
            else:
                outputs = torch.cat((outputs, b_output[0]), dim=0)

    np.save('features_{}_{}.npy'.format(MODEL_KEY, DATA_PATH), outputs.numpy())


if __name__ == '__main__':
    main()
