import pandas as pd

from features.add_article_unique_customers import add_article_unique_customers


def add_article_unique_customers_popularity(articles_df: pd.DataFrame, transactions_df: pd.DataFrame):
    """
    Calculates the top unique customers popularity of each article. (E.g. 1st most-unique customers, 2nd most-unique customers, etc.)
    :param articles_df: The articles dataframe.
    :param transactions_df: The transactions dataframe.
    :return: A column with the popularity of each article.
    """
    temp_articles_df = articles_df.copy()

    if 'unique_customers' not in temp_articles_df.columns:
        temp_articles_df = add_article_unique_customers(temp_articles_df, transactions_df)

    return temp_articles_df['unique_customers'].rank(method='dense', ascending=False).astype(int)
