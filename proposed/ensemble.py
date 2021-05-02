import pandas as pd
import numpy as np

method = 'average'

files = {
    0: 'exps/roberta-full-data/roberta-full-data.logits.csv',  # Roberta on full data
    1: 'exps/bert-full-data/bert-full-data.logits.csv',  # Bert on Full data
    2: 'exps/roberta-cust/roberta-cust.logits.csv',  # Roberta (custom merge)
    3: 'exps/roberta-seg-cust/roberta-seg-cust.logits.csv',  # Roberta (word segment + custom merge)
    4: 'exps/xlnet-full-data/xlnet-full-data.logits.csv'  # Xlnet
}


def run_ensemble():
    # confidence_files = [files[x] for x in [0, 1]] # This gives 90.22 on kaggle
    # confidence_files = [files[x] for x in [2, 3]] # This gives 90.78 on kaggle
    confidence_files = [files[x] for x in [1, 2, 3]] # This gives 90.86 on kaggle
    # confidence_files = [files[x] for x in [0, 1, 2, 3]] # This gives 90.66 on kaggle
    # confidence_files = [files[x] for x in [1, 2, 3, 4]]  # This gives 90.36 on kaggle

    TEST_SAMPLES = 10000
    cumm_prob0 = np.zeros((TEST_SAMPLES,))
    cumm_pred0 = np.zeros((TEST_SAMPLES,))
    for conf_file in confidence_files:
        df = pd.read_csv(conf_file)
        logits0 = np.array(df['Logits_0'])
        logits1 = np.array(df['Logits_1'])
        prob0 = np.exp(logits0) / (np.exp(logits0) + np.exp(logits1))
        cumm_prob0 = cumm_prob0 + prob0
        cumm_pred0 = cumm_pred0 + (logits0 > logits1) * 1.0
    expected_prob0 = cumm_prob0 / len(confidence_files)
    expected_pred0 = cumm_pred0 / len(confidence_files)

    ids = list(range(1, TEST_SAMPLES + 1))
    predictions = []
    for i in range(TEST_SAMPLES):
        if method == 'average':
            if expected_prob0[i] >= 0.5:
                predictions.append(-1)
            else:
                predictions.append(1)
        elif method == 'majority':
            if expected_pred0[i] >= 0.5:
                predictions.append(-1)
            else:
                predictions.append(1)

    outdf = pd.DataFrame()
    outdf['Id'] = ids
    outdf['Prediction'] = predictions
    outdf.to_csv('exps/submission.csv', index=False)
    print('Submission file available at: exps/submission.csv')
