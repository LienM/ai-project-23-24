import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def not_interacted_with_candidates(t, a, articles_col, k=10):
    '''
    Generate candidates for each customer that are the most popular items in categories that customer did not interact with.

    Parameters
    ----------
    t : pd.DataFrame
        Transactions DataFrame.
    a : pd.DataFrame
        Articles DataFrame.
    articles_col : str
        Name of column in articles DataFrame that contains categories.
        Column must be present in articles DataFrame.
    k : int, default 10
        Number of candidates to generate for each customer, by default 10.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, article_id, not_interacted_rank.
    '''
    
    # Get unique values of given category
    group_unique_values = a[articles_col].unique()
    group_df = pd.merge(t, a[['article_id', articles_col]])

    # Not interacted category for each customer
    not_interacted_with = group_df\
        .groupby('customer_id')[articles_col]\
        .apply(lambda x: np.array(list(set(x))))\
        .apply(lambda x: np.setdiff1d(group_unique_values, x))
    
    # Get k most popular articles in given category
    items_popularity = group_df\
        .groupby(articles_col)['article_id']\
        .value_counts()\
        .groupby(articles_col)\
        .head(k)\
        .reset_index()

    # Rank items by popularity (number of purchases)
    items_popularity['not_interacted_rank'] = items_popularity['count']\
        .rank(method='dense', ascending=False)\
        .astype('int16')
    
    items_popularity = items_popularity\
        .filter(items=['article_id', articles_col, 'not_interacted_rank'])\
        .sort_values(by=['not_interacted_rank'])

    candidates = []

    # For each customer get k most popular articles in categories that customer did not interact with
    for cid in tqdm(not_interacted_with.index.values):
        groups = not_interacted_with.loc[cid]

        cid_candidates = items_popularity\
            [items_popularity[articles_col].isin(groups)]\
            .head(k)\
            .drop(columns=[articles_col])
        
        cid_candidates['customer_id'] = cid

        candidates.append(cid_candidates)

    return pd.concat(candidates)[['customer_id', 'article_id', 'not_interacted_rank']]


def generate_seasonal_baskets(t, years, months, k, include_rank=False):
    '''
    Generate seasonal baskets for each customer. 
    Seasonal basket is purchase history of given customer in given months and years.

    Parameters
    ----------
    t : pd.DataFrame
        Transactions DataFrame.
    years : list
        List of years to include in seasonal baskets.
    months : list
        List of months to include in seasonal baskets.
    k : int
        Maximum number of items to generate for each customer.
    include_rank : bool, default False
        If True, include seasonal_basket_rank column in output DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, article_id, optionally: seasonal_basket_rank.
    '''
    t['year'] = t['t_dat'].dt.year

    t_seasonal = t[(t.t_dat.dt.year.isin(years)) & (t.t_dat.dt.month.isin(months))]\
        .groupby(['customer_id', 'year'])['article_id'].value_counts()\
        .groupby(['customer_id', 'year']).rank(method='dense', ascending=False) \
        .groupby(['customer_id', 'year']).head(k).astype('int16')\
        .reset_index()\
        .rename(columns={'count': 'seasonal_basket_rank'})

    if not include_rank:
        t_seasonal = t_seasonal.drop(columns=['seasonal_basket_rank'])
    
    return t_seasonal


def generate_seasonal_bestsellers(t, years, months, k, include_rank=False):
    '''
    Generate seasonal bestsellers. 
    Seasonal bestseller is an item that was purchased by the most customers in given months and years.
    Filtered by year and month, grouped by year.

    Parameters
    ----------
    t : pd.DataFrame
        Transactions DataFrame.
    years : list
        List of years to include in seasonal bestsellers.
    months : list
        List of months to include in seasonal bestsellers.
    k : int
        Maximum number of bestseller items to generate.
    include_rank : bool, default False
        If True, include seasonal_bestseller_rank column in output DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, article_id, optionally: seasonal_bestseller_rank.
    '''
    t['year'] = t['t_dat'].dt.year
    
    t_seasonal = t[(t.t_dat.dt.year.isin(years)) & (t.t_dat.dt.month.isin(months))]\
        .groupby(['year'])['article_id'].value_counts()\
        .groupby('year').rank(method='dense', ascending=False)\
        .groupby('year').head(k).astype('int16')\
        .reset_index()\
        .rename(columns={'count':'seasonal_bestseller_rank'})
    
    if not include_rank:
        t_seasonal = t_seasonal.drop(columns=['seasonal_bestseller_rank'])

    return t_seasonal


def generate_item_similarities(t, a, k=500):
    '''
    Generate item similarities using cosine similarity.

    Parameters
    ----------
    t : pd.DataFrame
        Transactions DataFrame.
    a : pd.DataFrame
        Articles DataFrame.
    k : int, default 500
        Number of most similar items to generate for each item.
    
    Returns
    -------
    dict
        Dictionary with article_id as key and pd.DataFrame with columns: article_id, similarity.
    '''
    # Prepare articles DataFrame for similarity calculation
    df = a.set_index('article_id')
    df['index_code'] = df['index_code'].cat.codes
    # Select only columns that are used in similarity calculation -- numerical columns
    df = df[['product_code', 'product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 'index_code', 'index_group_no', 'section_no', 'garment_group_no']]
    # Normalize columns
    df = (df - df.mean()) / df.std()

    # Calculate cosine similarities only for articles that are in transactions_train, to reduce the size of the matrix
    articles_ids = t.article_id.unique() 
    df = df[df.index.isin(articles_ids)]

    # For each item, get top k most similar items
    X = df.to_numpy()
    articles_arr = df.index.values
    sims = {}
    for aid, row in tqdm(zip(articles_arr, X)):
        similarities = cosine_similarity(row.reshape(1, -1), X)

        # Get top k most similar items and their similarity values
        top_n_sim = similarities.argsort()[:, -(k + 2):-1]
        top_n_sim_values = similarities[0, top_n_sim].reshape(-1)

        article_ids = articles_arr[top_n_sim].reshape(-1)
        sims[aid] = pd.DataFrame({'article_id': article_ids, 'similarity': top_n_sim_values})

    return sims


def generate_sim_not_bought_candidates(t, similarities_dict, k, include_score=False):
    '''
    Generate candidates for each customer that are the most similar items to items that customer did not interact with.

    Parameters
    ----------
    t : pd.DataFrame
        Transactions DataFrame.
    similarities_dict : dict
        Dictionary with article_id as key and pd.DataFrame with columns: article_id, similarity.
    k : int
        Number of candidates to generate for each customer.
    include_score : bool, default False
        If True, include similarity_score column in output DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: customer_id, article_id, optionally: similarity_score.
    '''

    # Group transactions by customer_id and create list of items that each customer bought
    user_purchases = t.groupby('customer_id')['article_id'].apply(list)

    total_candidates = []

    # For each customer, get k most similar items to items that customer did not interact with
    for cid, items in tqdm(user_purchases.items()): 

        sim_df = []
        for item in items:
            sim_df.append(similarities_dict[item])
        sim_df = pd.concat(sim_df)
        
        candidates = sim_df[~sim_df.article_id.isin(items)]\
            .drop_duplicates(subset=['article_id'], keep='first')\
            .sort_values(by='similarity', ascending=False)\
            .rename(columns={'similarity':'similarity_score'})\
            .head(k)
        
        candidates['customer_id'] = cid

        if not include_score:
            candidates.drop(columns=['similarity_score'], inplace=True)

        total_candidates.append(candidates)

    return pd.concat(total_candidates)