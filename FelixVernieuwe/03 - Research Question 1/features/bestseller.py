import pandas as pd


def weekly_bestseller_feature(transactions: pd.DataFrame, bestsellers_per_week_ranked: pd.DataFrame):
    """
    Add weekly bestseller feature to transactions
    :param transactions: Filtered transactions dataframe
    :param bestsellers_per_week_ranked: Top-k weekly bestseller products dataframe
    :return: transactions dataframe with weekly bestseller feature
    """

    # Note that bestsellers_per_week_ranked determines how high the rank is counted till we impute to 999
    transactions = pd.merge(transactions, bestsellers_per_week_ranked[['week', 'article_id', 'rank']],
                            on=['week', 'article_id'], how='left')
    transactions['rank'].fillna(999, inplace=True)
    transactions.rename(columns={'rank': 'weekly_rank'}, inplace=True)

    return transactions


def all_time_bestseller_feature(transactions: pd.DataFrame, most_purchased_articles: pd.DataFrame):
    """
    Add all time bestseller feature to transactions
    :param transactions: Filtered transactions dataframe
    :param most_purchased_articles: All time bestseller products dataframe
    :return: transactions dataframe with all time bestseller feature
    """

    transactions = pd.merge(transactions, most_purchased_articles[['article_id', 'rank']], on='article_id', how='left')
    transactions['rank'].fillna(999, inplace=True)
    transactions.rename(columns={'rank': 'all_time_rank'}, inplace=True)

    return transactions
