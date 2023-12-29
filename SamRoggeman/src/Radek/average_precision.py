"""
File: average_precision.py
Contains average precision at k and mean average precision at k functions created by Radek

"""
import numpy as np

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

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

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.

    This function computes the mean average prescision at k between two lists
    of items.

    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The mean average precision at k over the input lists

    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def calculate_apk(list_of_preds, list_of_gts):
    """
    Calculates the average precision at k for a list of predictions and a list of ground truths
    :param list_of_preds: list of lists of predicted elements (order matters in the lists)
    :param list_of_gts: list of lists of elements that are to be predicted (order doesn't matter)
    :return: score : double
            The average precision at k over the input lists
    """
    # for fast validation this can be changed to operate on dicts of {'cust_id_int': [art_id_int, ...]}
    # using 'data/val_week_purchases_by_cust.pkl'
    apks = []
    for preds, gt in zip(list_of_preds, list_of_gts):
        apks.append(apk(gt, preds, k=12))
    return np.mean(apks)



def calculate_apk_dicts(dict_of_preds, dict_of_gts):
    """
    Calculates the average precision at k for a dictionary of predictions and a dictionary of ground truths
    :param dict_of_preds: dict of {'cust_id_int': [art_id_int, ...]}
    :param dict_of_gts: dict of {'cust_id_int': [art_id_int, ...]}
    :return: score : double
    """
    apks = []
    for c_id, preds in dict_of_preds.items():
            gt = dict_of_gts[c_id]
            apks.append(apk(gt, preds, k=15))
    return np.mean(apks)