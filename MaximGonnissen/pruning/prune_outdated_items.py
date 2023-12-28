import pandas as pd


def prune_outdated_items(articles_df: pd.DataFrame, transactions_df: pd.DataFrame, cutoff_days: int = 365) -> (pd.DataFrame, pd.DataFrame):
    """
    Prunes items that have not been purchased in the last year.
    :param articles_df: The articles dataframe.
    :param transactions_df: The transactions dataframe.
    :param cutoff_days: The cutoff in days. --> If an item has not been purchased in the last [cutoff_days] days, it will be assumed no longer in stock.
    :return: The pruned articles and transactions dataframes.
    """
    transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])

    last_purchase_dates = transactions_df.groupby('article_id')['t_dat'].max().reset_index()

    outdated_items = last_purchase_dates[last_purchase_dates['t_dat'] < last_purchase_dates['t_dat'].max() - pd.Timedelta(days=cutoff_days)]['article_id']

    new_articles_df = articles_df[~articles_df['article_id'].isin(outdated_items)]
    new_transactions_df = transactions_df[~transactions_df['article_id'].isin(outdated_items)]

    new_articles_df = new_articles_df[new_articles_df['article_id'].isin(new_transactions_df['article_id'])]
    new_transactions_df = new_transactions_df[new_transactions_df['article_id'].isin(new_articles_df['article_id'])]

    return new_articles_df, new_transactions_df
