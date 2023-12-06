import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)

def hex_id_to_int(str):
    return int(str[-16:], 16)


def get_preds_and_actual(test, preds_col, test_week_transactions, sub, bestsellers_previous_week=None):
    '''
    Returns
        preds (list): list of lists of predictions of what customers bought in the test week.
        actual_purchases (list): list of lists of purchases that were actually made in the test week (including customers who didnt make a purchase).
        preds_dense (list): list of lists of predictions of what customers bought in the test week (only including customers who actually bought something in the test week).
        actual_purchases (list): list of lists of purchases that were actually made in the test week (only including customers who actually bought something in the test week).

    '''
    #if bestsellers_previous_week was provided, use it to pad the predictions when < 12 predictions are made
    if bestsellers_previous_week is not None:
        bestsellers_last_week = \
            bestsellers_previous_week[bestsellers_previous_week.week == bestsellers_previous_week.week.max()]['article_id'].tolist()

    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', preds_col], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()
    
    #Getting the list of items that were *actually* bought on the test week
    actual = test_week_transactions.sort_values(['customer_id'], ascending=False).groupby('customer_id')['article_id'].apply(list).to_dict()

    preds = []
    actual_purchases = []
    preds_dense = []
    actual_purchases_dense = []

    for c_id in customer_hex_id_to_int(sub.customer_id):
        purchase = actual.get(c_id, [])

        #getting lists of actual purchases and predictions (including non-purchasing customers)
        actual_purchases.append(purchase)
        pred = c_id2predicted_article_ids.get(c_id, [])
        if bestsellers_previous_week is not None:
            pred = pred + bestsellers_last_week
        preds.append(pred[:12])

        #getting lists of actual purchases and predictions (not including non-purchasing customers)
        if purchase:
            actual_purchases_dense.append(purchase)

            pred = c_id2predicted_article_ids.get(c_id, [])
            if bestsellers_previous_week is not None:
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



def get_evaluation_plots(test, pred_cols, test_week_transactions, bestsellers_previous_week, show_plots=True):
    '''
    Evaluates rankings (given as columns on test) by comparing to actual purchases made in the test week

    Parameters:
        test (pandas.DataFrame): dataframe containing all the samples that were used to test the rankers.
        pred_cols (list): list of names of column headers that correspond to different rankings for the given test samples.
        test_week_transactions (pandas.DataFrame): DataFrame containing all the transactions that *actually occured* during the test week.
        bestsellers_previous_week (pandas.DataFrame or None): dataframe containing the bestsellers and their bestseller rank from the previous week (used for padding the predictions if test doesnt contain enough - optional).
        show_plots (boolean): if true, creates matplotlib plots to visualise metrics for each of the prediction sets.

    Returns:
        mapk_per_col (dict): dictionary of lists of MAP@k scores for k from 1 to 12. Indexing done using elements of pred_cols as keys.
        mapk_per_col_dense (dict): same as mapk_per_col but scores computed only on customers who made a purchase in the test week.
        mrr_per_col (list): list of mrr scores for each of the given lists of rankings.
        mrr_per_col_dense (list): same as mrr_er_col but scores computed only on customers who made a purchase in the test week.
        pr_curves_dense (dict): dictionary where key is an element of pred_cols, and value is pr_curve (see get_pr_curve)
    '''
    mapk_per_col = {}
    mapk_per_col_dense = {}
    mrr_per_col = []
    mrr_per_col_dense = []
    pr_curves_dense = {}

    for pred_method in pred_cols:
        print(f"Evaluation for {pred_method}\n####################\n")
        #Getting predictions and actual purchases in same format
        preds, actual_purchases, preds_dense, actual_purchases_dense = get_preds_and_actual(test, pred_method, test_week_transactions, pd.read_csv('../../../Data/sample_submission.csv'), bestsellers_previous_week)

        mapk_per_col[pred_method] = [mapk(actual_purchases, preds, k=i) for i in range(1,12)]
        
        mapk_per_col_dense[pred_method] = [mapk(actual_purchases_dense, preds_dense, k=i) for i in range(1,12)]

        pr_curves_dense[pred_method] = get_pr_curve(actual_purchases_dense, preds_dense, 12)


        #MAPK of data at k=12 (competition requirements)
        print(f"MAP@12 (all customers):             {mapk_per_col[pred_method][-1]}")
        print(f"MAP@12 (only purchasing customers): {mapk_per_col_dense[pred_method][-1]}\n")

        mrr_per_col.append(mrr(actual_purchases, preds))
        mrr_per_col_dense.append(mrr(actual_purchases_dense, preds_dense))

        print(f"MRR (all customers):                {mrr_per_col[-1]}")
        print(f"MRR (only purchasing customers):    {mrr_per_col_dense[-1]}\n\n\n")

    

    #long winded plotting of metrics
    if show_plots:
        #visualising MAPK scores
        size = (10,6)
        plt.figure(figsize=size)
        for method, scores in mapk_per_col.items():
            plt.plot(range(1,12), scores, label=method)
        plt.ylabel("MAP score")
        plt.xlabel("k")
        plt.title(f"MAPK of predictions (all customers)")
        plt.legend()
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()

        plt.figure(figsize=size)
        for method, scores in mapk_per_col_dense.items():
            plt.plot(range(1,12), scores, label=method)
        plt.ylabel("MAP score")
        plt.xlabel("k")
        plt.title(f"MAPK of predictions (only purchasing customers)")
        plt.legend()
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


        #MRR for all customers
        plt.figure(figsize=size)
        plt.scatter(pred_cols, mrr_per_col)
        plt.ylabel("MRR score")
        plt.xlabel("Ranking Method")
        plt.title(f"MRR of predictions (all customers)")
        plt.xticks(rotation=45)
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


        #MRR for purchasing customers
        plt.figure(figsize=size)
        plt.scatter(pred_cols, mrr_per_col_dense)
        plt.ylabel("MRR score")
        plt.xlabel("Ranking Method")
        plt.title(f"MRR of predictions (only purchasing customers)")
        plt.xticks(rotation=45)
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


        #pr curve
        plt.figure(figsize=size)
        for method, scores in pr_curves_dense.items():
            plt.plot(scores[1], scores[0], label=method)
        for i in range(len(scores[0])):
            plt.annotate(f"k={i+1}", xy=(scores[1][i], scores[0][i]))
        plt.ylabel("Precision (avg. across customers)")
        plt.xlabel("Recall (avg. across customers)")
        plt.title(f"Precision-Recall curve up to k=12 (only customers who bought >= k items)")
        plt.legend()
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


        #precision@k
        plt.figure(figsize=size)
        for method, scores in pr_curves_dense.items():
            plt.plot(range(1, len(scores[0])+1), scores[0], label=method)
        plt.ylabel("Precision (avg. across customers)")
        plt.xlabel("k")
        plt.title(f"Precision@k (only customers who bought >= k items)")
        plt.legend()
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


        #recall@k
        plt.figure(figsize=size)
        for method, scores in pr_curves_dense.items():
            plt.plot(range(1, len(scores[1])+1), scores[1], label=method)
        plt.ylabel("Recall (avg. across customers)")
        plt.xlabel("k")
        plt.title(f"Recall@k (only customers who bought >= k items)")
        plt.legend()
        plt.grid(True, which="major", axis="y", alpha=0.5)
        plt.show()


    return mapk_per_col, mapk_per_col_dense, mrr_per_col, mrr_per_col_dense, pr_curves_dense