import pandas as pd


def all_time_rank_feature(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame):
    """
    Adds the all_time_rank feature to the transactions DataFrame.

    :param transactions: (filtered) transactions dataframe
    :param customers: Customer dataframe
    :param articles: Article dataframe
    :return: transactions, customers, articles
    """
    # Rank the articles based on the amount of times they have been purchased
    most_purchased_articles = transactions.groupby('article_id').size().sort_values(ascending=False)
    most_purchased_articles = most_purchased_articles.reset_index()
    most_purchased_articles['rank'] = most_purchased_articles.index + 1
    most_purchased_articles = most_purchased_articles.drop(columns=0)

    transactions['all_time_rank'] = transactions['article_id'].map(most_purchased_articles.set_index('article_id')['rank'])

    return transactions, customers, articles
