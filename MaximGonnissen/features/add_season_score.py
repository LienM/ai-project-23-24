import pandas as pd

from selection.sales_per_date import article_sales_per_date
from utils.season import Seasons


def add_season_scores(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the seasonal score for each article.
    :param transactions_df: DataFrame containing transactions.
    :return: DataFrame containing seasonal sales numbers for each article.
    """
    sales_per_date_df = article_sales_per_date(transactions_df)

    for season in Seasons.seasons:
        sales_per_date_df[season.season_name] = sales_per_date_df['t_dat'].apply(season.get_season_score).fillna(0)

    sales_per_date_df = sales_per_date_df.drop(columns=['t_dat'])

    sales_per_date_df = sales_per_date_df.groupby(['article_id']).sum().reset_index()
    sales_per_date_df = sales_per_date_df.drop(columns=['count'])

    return sales_per_date_df
