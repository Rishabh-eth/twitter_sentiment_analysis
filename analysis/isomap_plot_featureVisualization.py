import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.manifold import Isomap

# Folder features have the extracted features for both positive and negative training data corresponding to
# the models XLM-RoBERTa, BERT, XlNet and RoBERTa

output_path = 'output_plots/'

negative_examples = np.load('features/features_roberta_base_train_neg.txt.npy')

positive_examples = np.load('features/features_roberta_base_train_pos.txt.npy')

neg = negative_examples[10000:20000]
pos = positive_examples[10000:20000]

neg_t =StandardScaler().fit_transform(neg)
pos_t = StandardScaler().fit_transform(pos)

pca = Isomap(n_components=154, n_jobs = 4, n_neighbors = 5)
principalComponents_p = pca.fit_transform(pos_t)
principalComponents_n = pca.fit_transform(neg_t)

plt.clf()

plt.figure()
plt.figure(figsize=(12,12))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Component - 1',fontsize=20)
plt.ylabel('Component - 2',fontsize=20)
plt.title("ISOMAP Analysis on Twitter Dataset",fontsize=20)

for item in principalComponents_p:
    plt.scatter(item[0],item[1],c='b',s=10)
for item in principalComponents_n:
    plt.scatter(item[0],item[1],c='r',s=10)

filename = "roberta_iso.png"
plt.savefig(output_path + filename)
