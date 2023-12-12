import pandas as pd 
import numpy as np
import json
import os
from lightgbm import LGBMRanker


def prepare_data(t_df, bestsellers_prev_week, candidates, features, cols_to_use, test_week=105):
    '''
    Prepare data for training.

    Parameters
    ----------
    t_df : pd.DataFrame
        DataFrame with transactions.
    bestsellers_prev_week : pd.DataFrame
        DataFrame with bestsellers for previous week.
    candidates : list
        List of DataFrames with candidates.
    features : list
        List of DataFrames with features. DataFrames should have one at least but not all of following columns: week, article_id, customer_id.
    cols_to_use : list
        List of columns to use for training.
    test_week : int, default 105
        Week to use as test data. The default is 105.
    
    Returns
    -------
    train_X : pd.DataFrame
        Training data.
    train_y : pd.Series
        Training labels.
    test_X : pd.DataFrame
        Test data features.
    test : pd.DataFrame
        Test data.
    train_baskets : np.array
        Number of purchases for each customer week pair.    
    '''
    t_df['purchased'] = 1
    data = pd.concat([t_df] + candidates)
    data.purchased.fillna(0, inplace=True)
    data.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)

    print('Percentage of real transactions: ', data.purchased.mean())

    model_data = pd.merge(
        data,
        bestsellers_prev_week[['week', 'article_id', 'bestseller_rank']],
        on=['week', 'article_id'],
        how='left'
    )

    # Remove first week of data, as we don't have bestseller rank for it
    # (week was shifted by one) and fill missing values with 999 -- really bad rank
    model_data = model_data[model_data.week != model_data.week.min()]
    model_data.fillna({'bestseller_rank':999}, inplace=True)

    print('Mergining features...')
    for feature_df in features:

        if ('week' in feature_df.columns) and ('article_id' in feature_df.columns):
            model_data = pd.merge(
                model_data, 
                feature_df, 
                on=['week', 'article_id'], 
                how='left'
            )
            
        elif ('week' in feature_df.columns) and ('customer_id' in feature_df.columns):
            model_data = pd.merge(
                model_data, 
                feature_df, 
                on=['week', 'customer_id'], 
                how='left'
            )
            
        elif ('week' not in feature_df.columns) and ('article_id' in feature_df.columns):
            model_data = pd.merge(
                model_data, 
                feature_df, 
                on='article_id', 
                how='left'
            )
            
        elif ('week' not in feature_df.columns) and ('customer_id' in feature_df.columns):
            model_data = pd.merge(
                model_data, 
                feature_df, 
                on='customer_id', 
                how='left'
            )
    
    print('Done.')
    print('Sorting data...')
    model_data.sort_values(['week', 'customer_id'], inplace=True)
    model_data.reset_index(drop=True, inplace=True)
    print('Done.')
    print('Preparing for training...')
    train = model_data[model_data.week != test_week]
    test = model_data[model_data.week == test_week]\
        .drop_duplicates(['customer_id', 'article_id', 'sales_channel_id'])\
        .copy()
    # Basically how many purchased for each customer week pair -- so lgbm knows its one transaction
    train_baskets = train.groupby(['week', 'customer_id'])['article_id']\
        .count()\
        .values  
    
    train_X = train[cols_to_use]
    train_y = train['purchased']

    test_X = test[cols_to_use]

    assert test.purchased.mean() == 0, 'Test data should not contain any actual purchases!'

    print('Done.')

    return train_X, train_y, test_X, test, train_baskets


def train_model(train_X, train_y, train_baskets, params, cols_used, show_importance=None):
    '''
    Train LGBMRanker model for provided data. (output of prepare_data function)

    Parameters
    ----------
    train_X : pd.DataFrame
        Training data.
    train_y : pd.Series
        Training labels.
    train_baskets : np.array
        Number of purchases for each customer week pair.
    params : dict
        Parameters for LGBMRanker.
    cols_used : list
        List of columns used for training. (used for feature importance output)
    show_importance : int, optional
        Number of features to show in feature importance output. The default is None.

    Returns
    -------
    ranker : lightgbm.LGBMRanker
        Trained LGBMRanker model.
    '''

    ranker = LGBMRanker(**params)

    print('Training model...')

    ranker.fit(
        train_X, 
        train_y, 
        group=train_baskets
    )

    if show_importance:
        print('Feature importance:')
        for i in ranker.feature_importances_.argsort()[::-1][:show_importance]:
            print(cols_used[i], ranker.feature_importances_[i]/ranker.feature_importances_.sum())

    return ranker


def make_submission(c_df, test, test_X, ranker, submission_filler, submission_name):
    '''
    Create submission file and submit it to Kaggle via Kaggle API.

    Parameters
    ----------
    c_df : pd.DataFrame
        DataFrame with all customer ids.
    test : pd.DataFrame
        Test data.
    test_X : pd.DataFrame
        Test data features.
    ranker : lightgbm.LGBMRanker
        Trained LGBMRanker model.
    submission_filler : list
        List of articles to fill submission with. (if there are less than 12 recommendations)
    submission_name : str
        Name of the submission.
    
    Returns
    -------
    None.
    '''
    print('Starting submission process...')
    print('Calculating predictions...')
    test['preds'] = ranker.predict(test_X)
    c_id2predicted_article_ids = test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(lambda x: list(x)[:12])\
        .to_dict()
    print('Done.')

    print('Creating submission...')
    preds = []
    for c_id in c_df.customer_id:
        # c_id = idx_to_cust_id[c_id]  # map to original customer id
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + submission_filler
        preds.append(pred[:12])
    
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub = c_df[['customer_id']].copy()

    # This part is due to reduction mapping for customer ids

    with open('mappings/cust_id_to_idx.json', 'r') as f:
        cust_id_to_idx = json.load(f)

    idx_to_cust_id = {v: k for k, v in cust_id_to_idx.items()}

    sub['customer_id'] = sub['customer_id'].apply(lambda x: idx_to_cust_id[x])

    sub['prediction'] = preds
    print('Done.')
    
    print('Saving submission...')
    sub.to_csv(f'submissions/{submission_name}.csv.gz', index=False)
    print('Done.')

    cmd = f"kaggle competitions submit -c h-and-m-personalized-fashion-recommendations -f 'submissions/{submission_name}.csv.gz' -m {submission_name}"
    os.system(cmd)

    print('Submission saved and submitted to Kaggle.')







