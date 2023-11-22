import pandas as pd


def product_age_feature(all_transactions: pd.DataFrame, transactions: pd.DataFrame):
    """
    For every product, add the age (since first sighting) as a feature
    :param all_transactions: All transactions in the dataset
    :param transactions: Transactions in the training period
    :return: Transactions with 'age' feature added
    """

    # Get the first sighting of every article
    first_sighting = all_transactions.groupby('article_id')['week'].min().reset_index()

    # Adds the age feature to the data (age = 0 means the article was released in that week)
    # Merge transactions and first_sighting on article_id
    transactions = transactions.merge(first_sighting, on='article_id', how='left', suffixes=('', '_y'))

    # Calculate the age feature
    transactions['age'] = transactions['week'] - transactions['week_y']
    transactions.drop(columns=['week_y'], inplace=True)

    return transactions

