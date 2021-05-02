import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

files = {
    'XLNet': '../exps/xlnet-full-data/xlnet-full-data.csv',
    'SpacyCnn': '../exps/myspacy/myspacy.csv',
    'BERT': '../exps/bert-full-data/bert-full-data.logits.csv',
    'RoBERTa': '../exps/roberta-cust/roberta-cust.logits.csv',
    'Ensemble': '../exps/submission.csv'
}


def get_overlap(p1, p2):
    return np.mean((p1 == p2) * 1)*1


n = len(files)
M = np.zeros((n, n))
keys = list(files.keys())
for i in range(n):
    df1 = pd.read_csv(files[keys[i]])
    p1 = df1['Prediction'].to_numpy()
    for j in range(n):
        df2 = pd.read_csv(files[keys[j]])
        p2 = df2['Prediction'].to_numpy()
        overlap = get_overlap(p1, p2)
        M[i, j] = overlap

df = pd.DataFrame(M, index=keys, columns=keys)
fig, ax = plt.subplots(figsize=(7, 3))
sns_plot = sns.heatmap(df, annot=True, cmap='Blues', ax=ax)
fig.savefig("heatmap.png")
