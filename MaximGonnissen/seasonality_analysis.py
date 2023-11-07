import time
import multiprocessing as mp

from season import Season, seasons
from utils import DataFileNames, load_data_from_hnm, get_data_path
from progress_bar import ProgressBar

import pandas as pd


def get_season(date: pd.Timestamp) -> Season:
    """
    Returns the season for a given date.
    :param date: Date to get season for.
    :return: Season for given date.
    """
    for season in seasons:
        if season.in_season(date):
            return season


def get_season_score(season: Season, date: pd.Timestamp) -> float:
    """
    Returns the season score for a given date.
    :param season: Season to get score for.
    :param date: Date to get season score for.
    :return: Season score for given date for given season.
    """
    return season.get_season_score(date)


def article_sales_per_date(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates sales per date for each article.
    :param transactions_df: DataFrame containing transactions.
    :return: DataFrame containing sales per date for each article.
    """
    start_time = time.time()
    print('[ ] Calculating sales per date for each article...')

    transactions_df_trimmed = transactions_df.copy()
    transactions_df_trimmed = transactions_df_trimmed[['t_dat', 'article_id']]
    transactions_df_trimmed['t_dat'] = pd.to_datetime(transactions_df_trimmed['t_dat'])

    transactions_df_trimmed = transactions_df_trimmed.groupby(['article_id', 't_dat']).size().reset_index(name='count')

    print(f'[X] Calculated sales per date for each article in {time.time() - start_time:.2f} seconds.')
    return transactions_df_trimmed


def calculate_seasonal_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing sales per date for each article, calculates the seasonal sales numbers for each article.
    :param df: DataFrame containing sales per date for each article.
    :return: DataFrame containing seasonal sales numbers for each article.
    """
    start_time = time.time()
    print('[ ] Calculating seasonal sales numbers for each article...')

    new_df = df.copy()
    new_df['t_dat'] = pd.to_datetime(new_df['t_dat'])
    new_df['season'] = new_df['t_dat'].apply(get_season)
    new_df = new_df[['article_id', 'season', 'count']]

    new_df = new_df.groupby(['article_id', 'season']).sum().reset_index()

    print(f'[X] Calculated seasonal sales numbers for each article in {time.time() - start_time:.2f} seconds.')
    return new_df


def calculate_season_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing sales per date for each article, calculates the seasonal sales numbers for each article.
    :param df: DataFrame containing sales per date for each article.
    :return: DataFrame containing seasonal sales numbers for each article.
    """
    start_time = time.time()
    print('[ ] Calculating seasonal sales numbers for each article...')

    new_df = df.copy()
    new_df['t_dat'] = pd.to_datetime(new_df['t_dat'])

    with ProgressBar(seasons) as progress_bar:
        for season in progress_bar:
            new_df[season.season_name] = new_df['t_dat'].apply(lambda x: season.get_season_score(x))

    new_df = new_df.drop(columns=['t_dat'])

    new_df = new_df.groupby(['article_id']).sum().reset_index()
    new_df = new_df.drop(columns=['count'])

    print(f'[X] Calculated seasonal sales numbers for each article in {time.time() - start_time:.2f} seconds.')
    return new_df


if __name__ == '__main__':
    script_start_time = time.time()
    mp_pool_count = max(mp.cpu_count() - 1, 1)
    print(f'Using {mp_pool_count} cores for multiprocessing.')

    # Load data
    transactions_train = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN)

    # Calculate sales per date for each article
    article_sales_per_date_df = article_sales_per_date(transactions_train)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'article_sales_per_date.csv'
    article_sales_per_date_df.to_csv(output_path, index=False)

    # Calculate seasonal sales numbers for each article
    seasonal_sales_df = calculate_seasonal_sales(article_sales_per_date_df)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_sales.csv'
    seasonal_sales_df.to_csv(output_path, index=False)

    # Calculate seasonal scores for each article
    seasonal_scores_df = calculate_season_scores(article_sales_per_date_df)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_scores.csv'
    seasonal_scores_df.to_csv(output_path, index=False)
