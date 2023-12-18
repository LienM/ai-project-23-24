import pandas as pd

from features.add_article_total_sales import add_article_total_sales


def add_article_popularity(articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Calculates the popularity of each article. (E.g. 1st most popular, 2nd most popular, etc.)
    :param articles_df: The articles dataframe.
    :param transactions_df: The transactions dataframe.
    :return: A column with the popularity of each article.
    """
    temp_articles_df = articles_df.copy()

    if 'total_sales_count' not in temp_articles_df.columns:
        temp_articles_df = add_article_total_sales(temp_articles_df, transactions_df)

    return temp_articles_df['total_sales_count'].rank(method='dense', ascending=False).astype(int)
