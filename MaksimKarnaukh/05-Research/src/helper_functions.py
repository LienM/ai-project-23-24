from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


def read_parquet_datasets():

    basepath = '../../input/'
    transactions = pd.read_parquet(basepath + 'transactions_train.parquet')
    customers = pd.read_parquet(basepath + 'customers.parquet')
    articles = pd.read_parquet(basepath + 'articles.parquet')

    return transactions, customers, articles


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


# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def article_id_str_to_int(series):
    return series.astype('int32')


def article_id_int_to_str(series):
    return '0' + series.astype('str')


class Categorize(BaseEstimator, TransformerMixin):
    def __init__(self, min_examples=0):
        self.min_examples = min_examples
        self.categories = []

    def fit(self, X):
        for i in range(X.shape[1]):
            vc = X.iloc[:, i].value_counts()
            self.categories.append(vc[vc > self.min_examples].index.tolist())
        return self

    def transform(self, X):
        data = {X.columns[i]: pd.Categorical(X.iloc[:, i], categories=self.categories[i]).codes for i in
                range(X.shape[1])}
        return pd.DataFrame(data=data)


### Below is everything related to recall calculation ###

# Copied from NickWils https://github.com/LienM/ai-project-23-24/blob/main/NickWils/Lecture6/candidate-repurchase.ipynb
def recall(actual, predicted, k=12):
    if len(predicted) > k:
        predicted = predicted[:k]
    correct_predictions = [p for p in predicted if p in actual]
    return len(correct_predictions) / len(actual)
def recall12(actual, predicted, k=12):
    return np.mean([recall(a, p, k) for a, p in zip(actual, predicted)])
#

def calculateRecall(expected, retrieved):
    """
    R(ecall) = TP/(TP+FN)
    :param expected: list of expected values
    :param retrieved: list of retrieved values
    :return:
    """
    # number of retrieved values that are also in expected (True positive)
    TP = len([ret for ret in retrieved if ret in expected])
    # number of expected values that aren't retrieved (False negative)
    FN = len([ex for ex in expected if ex not in retrieved])
    # recall calculation (by formula)
    recall = TP / (TP + FN)
    return recall


def mean_recall(expected, retrieved):
    """
    Calculate mean recall for all users
    :param expected: list of expected values
    :param retrieved: list of retrieved values
    :return:
    """
    recalls = [calculateRecall(ex, ret) for ex, ret in zip(expected, retrieved)]
    mean_recall = np.mean(recalls)
    return mean_recall


def calculate_recall_per_customer_batch(validation, top_candidates_3feat_prev_week, customer_batch, top_x_age=25):

    validation_corresp_customers = validation[validation['customer_id'].isin(customer_batch)]

    # Get the corresponding candidates generated for the  customers in the last week
    candidates_last_week = top_candidates_3feat_prev_week[
        (top_candidates_3feat_prev_week['week'] == validation_corresp_customers['week'].max()) &
        (top_candidates_3feat_prev_week['customer_id'].isin(validation_corresp_customers['customer_id'].unique()))
    ]

    validation_corresp_customers = validation_corresp_customers.sort_values(['customer_id', 'article_id'])
    candidates_last_week = candidates_last_week.sort_values(['customer_id', 'article_id'])

    # if validation['customer_id'].nunique() * top_x_age == candidates_last_week.shape[0]:
    #     print("Validation and candidates_last_week have the same number of unique customers (OKAY).")
    # else:
    #     print("Validation and candidates_last_week don't have the same number of unique customers (NOT REALLY OKAY).")

    # Group purchases and candidates by customer_id
    actual_purchases_last_week = validation_corresp_customers.groupby('customer_id')['article_id'].apply(list)
    predicted_candidates_last_week = candidates_last_week.groupby('customer_id')['article_id'].apply(list)

    # Calculate recall between actual purchases and predicted candidates for the last week
    recall_last_week = mean_recall(actual_purchases_last_week, predicted_candidates_last_week)

    print("Recall Score on Candidates for Last Week:", recall_last_week)

    return recall_last_week


def calculate_recall_per_week(validation, top_candidates_3feat_prev_week, customer_batch, amount_of_weeks=5, top_x_age=25):

    overall_mean_recalls = {}

    for week in range(validation['week'].max(), validation['week'].max() - amount_of_weeks, -1):

        # Filter validation and candidates for the current week
        validation_week = validation[validation['week'] == week]
        validation_corresp_customers = validation_week[validation_week['customer_id'].isin(customer_batch)]

        candidates_last_week = top_candidates_3feat_prev_week[
            (top_candidates_3feat_prev_week['week'] == week) &
            (top_candidates_3feat_prev_week['customer_id'].isin(validation_corresp_customers['customer_id'].unique()))
        ]

        validation_corresp_customers = validation_corresp_customers.sort_values(['customer_id', 'article_id'])
        candidates_last_week = candidates_last_week.sort_values(['customer_id', 'article_id'])

        # Group purchases and candidates by customer_id
        actual_purchases_week = validation_corresp_customers.groupby('customer_id')['article_id'].apply(list)
        predicted_candidates_week = candidates_last_week.groupby('customer_id')['article_id'].apply(list)

        # Calculate recall between actual purchases and predicted candidates for the current week
        recall_week = mean_recall(actual_purchases_week, predicted_candidates_week)
        overall_mean_recalls[week] = recall_week

        print(f"Week {week}: Recall Score on Candidates for Last Week: {recall_week}")

    return overall_mean_recalls
