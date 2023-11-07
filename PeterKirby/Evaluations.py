import numpy as np

def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)

def hex_id_to_int(str):
    return int(str[-16:], 16)


def get_preds_and_actual(test, preds_col, test_week_transactions, bestsellers_previous_week, sub):
    '''
    Returns
        preds (list): list of lists of predictions of what customers bought in the test week.
        actual_purchases (list): list of lists of purchases that were actually made in the test week (including customers who didnt make a purchase).
        preds_dense (list): list of lists of predictions of what customers bought in the test week (only including customers who actually bought something in the test week).
        actual_purchases (list): list of lists of purchases that were actually made in the test week (only including customers who actually bought something in the test week).

    '''
    bestsellers_last_week = \
        bestsellers_previous_week[bestsellers_previous_week.week == bestsellers_previous_week.week.max()]['article_id'].tolist()

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', preds_col], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()
    
    #Getting the list of items that were *actually* bought on the test week
    actual = test_week_transactions.sort_values(['customer_id'], ascending=False).groupby('customer_id')['article_id'].apply(list).to_dict()

    #getting lists of actual purchases and predictions (including non-purchasing customers)
    preds = []
    actual_purchases = []

    for c_id in customer_hex_id_to_int(sub.customer_id):
        actual_purchases.append(actual.get(c_id, []))
        
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + bestsellers_last_week
        preds.append(pred[:12])

    #getting lists of actual purchases and predictions (not including non-purchasing customers)
    preds_dense = []
    actual_purchases_dense = []

    for c_id in customer_hex_id_to_int(sub.customer_id):
        purchase = actual.get(c_id, [])
        if purchase:
            actual_purchases_dense.append(purchase)

            pred = c_id2predicted_article_ids.get(c_id, [])
            pred = pred + bestsellers_last_week
            preds_dense.append(pred[:12])

    return preds, actual_purchases, preds_dense, actual_purchases_dense




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
    of lists of items.

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




def mrr(actual, predicted):
    '''
    Computes the mean reciprocal rank of the given predictions according to the given actual lists.
    '''

    #for each customer,
    #   for each predicted item:
    #       if the item is in the actual, add the reciprocal of the rank.

    acc = 0

    for cust in range(len(predicted)):
        for rank in range(len(predicted[cust])):
            if predicted[cust][rank] in actual[cust]:
                    acc += 1/(rank+1)
                    break
            
    return acc/len(actual)


def precision_recall_at_k(actual, predicted, k):
    '''
    Returns
        precision (float): single value of precision@k (proportion of correct predictions out of all predicted) (not average of all ranks up to k).
        recall (float): single value of recall@k (proportion of correct predictions out of all correct).
    '''
    if len(actual) < k:
        return np.nan, np.nan

    predicted_trunc = predicted[:k]
    tp = 0

    for i,p in enumerate(predicted_trunc):
        if p in actual and p not in predicted_trunc[:i]:
            tp += 1.0

    precision = tp/k
    recall = tp/len(actual)
    return precision, recall



def get_pr_curve(actual, predicted, max_k):
    '''
    Returns
        pr_curve (numpy.array): a 2d array of shape (2, max_k). First sub-array containing precision values, second containing recall values.
                                Second dimension is the k value-1. E.g. precision@3 would be pr_curve[0][2]
    '''

    pr_curve = np.empty((2, max_k))


    for k in range(max_k):
        precision_recalls = np.array([precision_recall_at_k(a,p,k+1) for a,p in zip(actual, predicted)])
        pr_curve[0,k], pr_curve[1,k] = np.nanmean(precision_recalls, axis=0)

    return pr_curve

