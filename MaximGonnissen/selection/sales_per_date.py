import pandas as pd


def article_sales_per_date(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sales per date for each article.
    :param transactions_df: DataFrame containing transactions.
    :return: DataFrame containing sales per date for each article.
    """
    transactions_df_trimmed = transactions_df.copy()[['t_dat', 'article_id']]
    transactions_df_trimmed['t_dat'] = pd.to_datetime(transactions_df_trimmed['t_dat'])

    transactions_df_trimmed = transactions_df_trimmed.groupby(['article_id', 't_dat']).size().reset_index(name='count')

    return transactions_df_trimmed
