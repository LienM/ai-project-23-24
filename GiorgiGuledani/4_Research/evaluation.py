import numpy as np
import pandas as pd
from utils import path

def apk(actual, predicted, k=12):
    """
    source: https://raw.githubusercontent.com/benhamner/Metrics/master/Python/ml_metrics/average_precision.py

    Computes the average precision at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def eval_sub_MAP(sub_csv, skip_cust_with_no_purchases=True):
    """
    source: Radek

    evaluates the mean average precision at k (MAP@12) given a submission file using MAP@12.
    Uses validation set made of transactions of the last week per customer.

    """
    sub=pd.read_csv(sub_csv)
    validation_set = pd.read_parquet(f'{path}/validation_ground_truth.parquet')

    apks = []

    no_purchases_pattern = []
    for prediction, ground_truth in zip(sub.prediction.str.split(), validation_set.prediction.str.split()):
        if skip_cust_with_no_purchases and (ground_truth == no_purchases_pattern): continue
        apks.append(apk(ground_truth,ground_truth, k=12))
    return np.mean(apks)
