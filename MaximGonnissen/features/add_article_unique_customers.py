import pandas as pd


def add_article_unique_customers(articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Calculates the amount of unique customers that bought each article.
    :param articles_df: Articles dataframe to use for adding features.
    :param transactions_df: Transactions dataframe to use for adding features.
    :return: Articles dataframe with added features.
    """
    temp_articles_df = articles_df.copy()

    sales_count = transactions_df.groupby('article_id')['customer_id'].nunique().reset_index(name='unique_customers')

    temp_articles_df = temp_articles_df.merge(sales_count, on='article_id', how='left')

    return temp_articles_df
