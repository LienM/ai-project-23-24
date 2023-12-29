import pandas as pd

from selection.sales_per_date import article_sales_per_date
from utils.season import Seasons


def add_in_season_sales(transactions_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing sales per date for each article, calculates the seasonal sales numbers for each article.
    :param transactions_df: DataFrame containing sales per date for each article.
    :param articles_df: Articles DataFrame.
    :return: Articles DataFrame with seasonal sales numbers added.
    """
    seasonal_sales_df = article_sales_per_date(transactions_df)

    seasonal_sales_df['season'] = seasonal_sales_df['t_dat'].apply(Seasons.get_season)
    seasonal_sales_df = seasonal_sales_df[['article_id', 'season', 'count']]

    seasonal_sales_df = seasonal_sales_df.groupby(['article_id', 'season']).sum().reset_index()

    seasonal_sales_df = seasonal_sales_df.pivot(index='article_id', columns='season', values='count').reset_index()

    return articles_df.merge(seasonal_sales_df, on='article_id', how='left')
