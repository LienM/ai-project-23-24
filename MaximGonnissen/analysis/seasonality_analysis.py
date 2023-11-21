import datetime
import time
from pathlib import Path
from typing import Union
from io import BytesIO

from utils.season import Season, seasons
from utils.utils import DataFileNames, load_data_from_hnm, get_data_path, load_data, ProjectConfig
from utils.progress_bar import ProgressBar

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


def article_sales_per_date(transactions_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Calculates sales per date for each article.
    :param transactions_df: DataFrame containing transactions.
    :param verbose: Whether to print progress.
    :return: DataFrame containing sales per date for each article.
    """
    start_time = time.time()
    if verbose:
        print('[ ] Calculating sales per date for each article...')

    transactions_df_trimmed = transactions_df.copy()
    transactions_df_trimmed = transactions_df_trimmed[['t_dat', 'article_id']]
    transactions_df_trimmed['t_dat'] = pd.to_datetime(transactions_df_trimmed['t_dat'])

    transactions_df_trimmed = transactions_df_trimmed.groupby(['article_id', 't_dat']).size().reset_index(name='count')

    if verbose:
        print(f'[X] Calculated sales per date for each article in {time.time() - start_time:.2f} seconds.')

    return transactions_df_trimmed


def calculate_seasonal_sales(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Given a DataFrame containing sales per date for each article, calculates the seasonal sales numbers for each article.
    :param df: DataFrame containing sales per date for each article.
    :param verbose: Whether to print progress.
    :return: DataFrame containing seasonal sales numbers for each article.
    """
    start_time = time.time()
    if verbose:
        print('[ ] Calculating seasonal sales numbers for each article...')

    new_df = df.copy()
    new_df['t_dat'] = pd.to_datetime(new_df['t_dat'])
    new_df['season'] = new_df['t_dat'].apply(get_season)
    new_df = new_df[['article_id', 'season', 'count']]

    new_df = new_df.groupby(['article_id', 'season']).sum().reset_index()

    if verbose:
        print(f'[X] Calculated seasonal sales numbers for each article in {time.time() - start_time:.2f} seconds.')

    return new_df


def calculate_season_scores(df: pd.DataFrame, max_score_offset: int, max_score_day_range: int, verbose: bool = True) -> pd.DataFrame:
    """
    Given a DataFrame containing sales per date for each article, calculates the seasonal sales numbers for each article.
    :param df: DataFrame containing sales per date for each article.
    :param max_score_offset: Offset from start of season to max score day.
    :param max_score_day_range: Range of days around max score day to calculate score for.
    :param verbose: Whether to print progress.
    :return: DataFrame containing seasonal sales numbers for each article.
    """
    start_time = time.time()
    if verbose:
        print('[ ] Calculating seasonal scores for each article...')

    new_df = df.copy()
    new_df['t_dat'] = pd.to_datetime(new_df['t_dat'])

    with ProgressBar(seasons, show=verbose) as progress_bar:
        for season in progress_bar:
            new_df[season.season_name] = new_df['t_dat'].apply(
                lambda x: season.get_season_score(x, max_score_offset, max_score_day_range))

    new_df = new_df.drop(columns=['t_dat'])

    new_df = new_df.groupby(['article_id']).sum().reset_index()
    new_df = new_df.drop(columns=['count'])

    if verbose:
        print(f'[X] Calculated seasonal scores for each article in {time.time() - start_time:.2f} seconds.')

    return new_df


def calculate_top_sales(df: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime,
                        top_x: int = 12) -> pd.DataFrame:
    """
    Calculate best-selling seasonal items
    :param df: Dataframe with season scores per item
    :param start_date: Date to start from
    :param end_date: Date to end at
    :param top_x: Calculate top x best-selling items
    :return: Dataframe with top x best-selling items for each day
    """
    out_df = pd.DataFrame(columns=['date', 'items'])
    out_df['date'] = pd.to_datetime(out_df['date'])

    for day in pd.date_range(start_date, end_date):
        season = get_season(day)
        top_x_items = df.sort_values(by=season.season_name, ascending=False).head(top_x)['article_id'].tolist()
        out_df = out_df._append({'date': day, 'items': ' '.join([str(item_id) for item_id in top_x_items])},
                                ignore_index=True)

    return out_df


def predict_top_items(df: pd.DataFrame) -> list:
    """
    Predict most popular items in the period
    :param df: Dataframe with top items per day based on season scores
    :return: List of top items
    """
    # Go through each row, items are already sorted based on season scores
    # Each item gets a score based on the position in the list (leftmost = highest score)
    # The item with the highest score is the most popular item
    # Make a final selection of the top X items, where X is the length of the item column in the original dataframe

    item_scores = {}
    for index, row in df.iterrows():
        items = row['items'].split(' ')
        for i in range(len(items)):
            item_scores[items[i]] = item_scores.get(items[i], 0) + len(items) - i

    return sorted(item_scores, key=item_scores.get, reverse=True)[:len(df['items'].iloc[0].split(' '))]


def run_seasonal_analysis(max_score_offset: int, max_score_day_range: int, rerun_seasonal_scores: bool,
                          to_csv: bool = True, verbose: bool = True, submission_suffix: str = None) -> Union[Path, BytesIO]:
    # Output paths
    seasonal_scores_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_scores.csv'
    article_sales_per_date_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'article_sales_per_date.csv'
    seasonal_sales_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_sales.csv'
    top_seasonal_sales_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'top_seasonal_sales.csv'

    if verbose:
        print(f"Script started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
        print(
            f"Using season parameters: max_score_offset={max_score_offset}, max_score_day_range={max_score_day_range}.")

    article_sales_per_date_df = None
    if not article_sales_per_date_path.exists():
        transactions_train = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN, verbose, dtype={'article_id': str})

        # Calculate sales per date for each article
        article_sales_per_date_df = article_sales_per_date(transactions_train)

        if to_csv:
            article_sales_per_date_df.to_csv(article_sales_per_date_path, index=False)
    else:
        article_sales_per_date_df = load_data(article_sales_per_date_path, verbose, dtype={'article_id': str})

    seasonal_sales_df = None
    if not seasonal_sales_path.exists():
        # Calculate seasonal sales numbers for each article
        seasonal_sales_df = calculate_seasonal_sales(article_sales_per_date_df)

        if to_csv:
            seasonal_sales_df.to_csv(seasonal_sales_path, index=False)
    else:
        seasonal_sales_df = load_data(seasonal_sales_path, verbose, dtype={'article_id': str})

    seasonal_scores_df = None
    if not seasonal_scores_path.exists() or rerun_seasonal_scores:
        # Calculate seasonal scores for each article
        seasonal_scores_df = calculate_season_scores(article_sales_per_date_df, max_score_offset, max_score_day_range, verbose=verbose)

        if to_csv:
            seasonal_scores_df.to_csv(seasonal_scores_path, index=False)
    else:
        seasonal_scores_df = load_data(seasonal_scores_path, verbose, dtype={'article_id': str})

    top_seasonal_sales_df = None
    if not top_seasonal_sales_path.exists() or rerun_seasonal_scores:
        # Calculate top seasonal sales
        top_seasonal_sales_df = calculate_top_sales(seasonal_scores_df, ProjectConfig.DATA_END,
                                                    ProjectConfig.DATA_END + datetime.timedelta(days=7))

        if to_csv:
            top_seasonal_sales_df.to_csv(top_seasonal_sales_path, index=False)
    else:
        top_seasonal_sales_df = load_data(top_seasonal_sales_path, verbose, dtype={'article_id': str})

    top_items = predict_top_items(top_seasonal_sales_df)
    top_items_string = ' '.join([str(item_id) for item_id in top_items])

    submission_df = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION, verbose)
    submission_df['prediction'] = top_items_string

    output = None

    if to_csv:
        output = get_data_path() / DataFileNames.OUTPUT_DIR / ('submission' + (
            f'_{submission_suffix}' if submission_suffix else '') + '.csv')
    else:
        output = BytesIO()

    submission_df.to_csv(output, index=False)

    return output


if __name__ == '__main__':
    run_seasonal_analysis(-30, 30, True, to_csv=True, verbose=False, submission_suffix='test')
