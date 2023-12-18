import pandas as pd


def add_article_total_sales(articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Calculates the total sales count of each article.
    :param articles_df: Articles dataframe to use for adding features.
    :param transactions_df: Transactions dataframe to use for adding features.
    :return: Articles dataframe with total sales count
    """
    temp_articles_df = articles_df.copy()

    sales_count = transactions_df.groupby('article_id').size().reset_index(name='total_sales_count')

    temp_articles_df = temp_articles_df.merge(sales_count, on='article_id', how='left')

    return temp_articles_df
