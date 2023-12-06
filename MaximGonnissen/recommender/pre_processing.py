from pruning.prune_inactive import prune_inactive
from pruning.prune_no_purchases import prune_no_purchases
from pruning.prune_outdated_items import prune_outdated_items

from features.add_age_group import add_age_group
from features.add_gender import add_gender, add_gender_scores_to_articles

import pandas as pd


def pre_process(articles_df: pd.DataFrame, customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Pre-process our dataframes.
    :param articles_df: Articles dataframe.
    :param customers_df: Customers dataframe.
    :param transactions_df: Transactions dataframe.
    :return: (articles_df, customers_df, transactions_df)
    """

    # Prune dataframes
    customers_df = prune_inactive(customers_df)
    customers_df = prune_no_purchases(customers_df, transactions_df)

    articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df)

    # Add features
    articles_df['gender_score'] = add_gender_scores_to_articles(articles_df)

    customers_df['age_group'] = add_age_group(customers_df)
    customers_df['gender'] = add_gender(customers_df, transactions_df, articles_df)

    return articles_df, customers_df, transactions_df
