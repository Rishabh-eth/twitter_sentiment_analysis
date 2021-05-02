import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import XLNetModel
from transformers import XLNetTokenizer

MODEL_KEY = 'xlnet-base-cased'
MODEL_WEIGHTS_PATH = 'weights_1.pth'
DATA_PATH = ['train_pos.txt', 'train_neg.txt']
BATCH_SIZE = 32
MAX_LEN = 140


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("PyTorch Device: {}".format(device))


def get_feature_extractor_model(model_name, model_weight_path):
    print("Loading Model Architecture...")
    model = XLNetModel.from_pretrained(model_name)

    # Set both the models to evaluation mode
    model.eval()

    # Make sure that weights are indeed the same in both the model
    return model


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
    tokenizer = XLNetTokenizer.from_pretrained(MODEL_KEY)
    model.cuda()

    for path in DATA_PATH:
        print("starting", path)
        sentences = read_data(data_path=path)
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

                # for xlnet take last hidden state as summary
                # https://github.com/huggingface/transformers/blob/32883b310ba30d72e67bb2ebb5847888f03a90a8/src/transformers/modeling_utils.py#L1154
                hidden_state_xlnet = b_output[0][:,-1]
                if outputs is None:
                    outputs = hidden_state_xlnet
                else:
                    outputs = torch.cat((outputs, hidden_state_xlnet), dim=0)

        np.save('features_{}_{}.npy'.format("xlnet_base", path), outputs.cpu().numpy())
        print("done", path)

if __name__ == '__main__':
    main()
