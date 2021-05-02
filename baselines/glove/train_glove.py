#Load necessary library functions

import numpy as np
import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

# Directory

GLOVE_DIRECTORY="glove.twitter.27B/"
GLOVE_FILE="glove.twitter.27B.200d.txt"
input_path = 'twitter-datasets/' 
ouput_path = 'output/'

#Loading the data

positive_examples = list(open(input_path + "train_pos_full.txt", "r").readlines())
positive_examples = [s.strip() for s in positive_examples] 
negative_examples = list(open(input_path + "train_neg_full.txt", "r").readlines())
negative_examples = [s.strip() for s in negative_examples]
x = positive_examples + negative_examples
positive_labels = [1 for _ in positive_examples]
negative_labels = [0 for _ in negative_examples]
y = np.concatenate([positive_labels, negative_labels], 0)
shuffle_indices = np.random.permutation(np.arange(len(y)))
X_train= np.array(x)[shuffle_indices]
Y_train = np.array(y)[shuffle_indices]

# train test split

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=37)

# get embeddings

embeddings = {}
with open(os.path.join(GLOVE_DIRECTORY, GLOVE_FILE), 'r',encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs

a_file = open(output_path + "embeddings.pkl", "wb")
pickle.dump(embeddings, a_file)
a_file.close()

# training data for each message, taking the average

line_train_embeddings = []
for line in x_train:
    line_emb = np.zeros(200)
    fline_emb=[]
    length = len(line)
    for word in line.strip().split():
        tmp=[]
        if word in embeddings.keys():
            for (item1, item2) in zip(line_emb, embeddings[word]):
                tmp.append(item1+item2)
            line_emb=tmp        
    for item in line_emb:
        fline_emb.append(item/length)
    line_train_embeddings.append(fline_emb)
b_file = open(output_path + "train.pkl", "wb")
pickle.dump(line_train_embeddings, b_file)
b_file.close()

line_test_embeddings = []
for line in x_test:
    line_emb = np.zeros(200)
    fline_emb=[]
    length = len(line)
    for word in line.strip().split():
        tmp=[]
        if word in embeddings.keys():
            for (item1, item2) in zip(line_emb, embeddings[word]):
                tmp.append(item1+item2)
            line_emb=tmp        
    for item in line_emb:
        fline_emb.append(item/length)
    line_test_embeddings.append(fline_emb)
c_file = open(output_path + "valid.pkl", "wb")
pickle.dump(line_test_embeddings, c_file)
c_file.close()

# Final submission embedding

final_examples = list(open(input_path + "test_data.txt", "r").readlines())
final_test_embeddings = []
for line in final_examples:
    line_emb = np.zeros(200)
    fline_emb=[]
    length = len(line)
    for word in line.strip().split():
        tmp=[]
        if word in embeddings.keys():
            for (item1, item2) in zip(line_emb, embeddings[word]):
                tmp.append(item1+item2)
            line_emb=tmp        
    for item in line_emb:
        fline_emb.append(item/length)
    final_test_embeddings.append(fline_emb)

c_file = open(output_path + "test.pkl", "wb")
pickle.dump(final_test_embeddings, c_file)
c_file.close()


scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=8, penalty='l2',max_iter=5000)
model=scikit_log_reg.fit(line_train_embeddings,y_train)

filename = 'finalized_model.sav'
pickle.dump(model, open(output_path + filename, 'wb'))












