import pandas as pd

def candidate_new_arrivals(candidate_customers: pd.DataFrame, all_transactions: pd.DataFrame, weekly_purchase_price: pd.DataFrame, k=5, max_age=0):
    """
    For every customer candidate, add the new arrivals of the week as potential candidates
    :param candidate_customers: Customers that can receive candidates for every week in given period
    :param all_transactions: Transactions in the entire dataset (used for finding relative age of product)
    :param k: Number of new arrivals to add as candidates
    :param max_age: Maximum age of new arrivals to add as candidates
    :return:
    """

    # Get the first sighting of every article
    first_sighting = all_transactions.groupby('article_id')['week'].min().reset_index()

    # Adds the age feature to the data (age = 0 means the article was released in that week)
    # all_transactions['age'] = all_transactions['week'] - all_transactions['article_id'].map(first_sighting)

    all_transactions = all_transactions.merge(first_sighting, on="article_id", how='left', suffixes=('', '_y'))
    all_transactions['age'] = all_transactions['week'] - all_transactions['week_y']
    all_transactions.drop(columns=['week_y'], inplace=True)


    # New arrivals are all product whose age is equal to or below the max_age in the data
    # new_arrivals = all_transactions[all_transactions['age'] == 0][['week', 'article_id']].drop_duplicates().reset_index(drop=True)
    new_arrivals = all_transactions[all_transactions['age'] <= max_age][['week', 'article_id']].drop_duplicates().reset_index(drop=True)

    # For every week, only keep the k newest arrivals
    new_arrivals = new_arrivals.groupby(['week']).head(k)

    # For every customer, add the new arrivals of the week as a candidate
    new_arrival_candidates = pd.merge(candidate_customers, new_arrivals, on='week')
    new_arrival_candidates = new_arrival_candidates.merge(weekly_purchase_price, on=['week', 'article_id'], how='left')

    return new_arrival_candidates
