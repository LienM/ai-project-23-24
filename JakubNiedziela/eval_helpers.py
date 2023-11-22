# This file is simply an aggregation of functions from this notebook: 
# https://www.kaggle.com/code/jniedziela/understanding-the-evaluation-metric/edit
# Thanks to this we can easily evaluate prediction using Mean Average Precision


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_recall_score(test_week_transactions, predictions_df, k=100):
    y_true = test_week_transactions.groupby('customer_id')['article_id'].apply(list).reset_index()
    y_true.columns = ['customer_id', 'y_true']
    predictions_df.columns = ['customer_id', 'y_pred']
    eval_df = pd.merge(y_true, predictions_df, on='customer_id')
    return recall_at_k(eval_df, k)


def recall_at_k(df, k):
    """
    Calculate the average Recall@K from a DataFrame using vectorized operations.

    Parameters:
    df (DataFrame): A DataFrame with columns 'customer_id', 'actual_bought', and 'candidates'.
    k (int): The number of top recommendations to consider for each user.

    Returns:
    float: The average Recall@K across all users.
    """
    def calculate_recall(row):
        top_k_items = set(row['y_pred'][:k])
        actual_items = set(row['y_true'])
        hits = len(top_k_items.intersection(actual_items))
        return hits / len(actual_items) if actual_items else 0

    recall_scores = df.apply(calculate_recall, axis=1)
    return recall_scores.mean()


def plot_recall_by_k(df, legend_column):
    '''
    Plot Recall@K for different values of K.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: customer_id, y_true, y_pred
    column : str
        Name of the column for plot label.

    Returns
    -------
        None.
    '''
    if not {'k', legend_column, 'recall_score'}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain 'k', '{legend_column}', and 'recall_score' columns")

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='k', y='recall_score', hue=legend_column)
    plt.ylim(0, 0.06)
    plt.title('Recall Scores by k')
    plt.xlabel('k Parameter')
    plt.ylabel('Recall Score')
    plt.legend(title='Strategy Name')
    plt.grid(True)
    plt.show()


def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k


def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0
    

def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        
    return ap / min(k, len(y_true))


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])