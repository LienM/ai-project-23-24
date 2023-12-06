import numpy as np
import pandas as pd
from utils import path

def apk(actual, predicted, k):
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

def recall(actual, predicted, k):
    """
    Computes the recall at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0

    if not actual:
        return 0.0

    return num_hits / min(len(actual), k)


def eval_sub(sub_csv, method, skip_cust_with_no_purchases=True, k=12):
    """
    source: Radek

    evaluates the mean average precision at k (MAP@12) given a submission file using MAP@12.
    Uses validation set made of transactions of the last week per customer.

    """
    sub=pd.read_csv(sub_csv)
    validation_set = pd.read_parquet(f'{path}/validation_ground_truth.parquet')

    results = []

    no_purchases_pattern = []
    for prediction, ground_truth in zip(sub.prediction.str.split(), validation_set.prediction.str.split()):
        if skip_cust_with_no_purchases and (ground_truth == no_purchases_pattern): continue
        if method == "map":
            results.append(apk(ground_truth,prediction, k))
        elif method == "recall":
            results.append(recall(ground_truth, prediction, k))

    if method == "map":
        return np.mean(results)
    elif method == "recall":
        return np.sum(results) / len(sub)