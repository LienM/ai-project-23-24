import datetime
import json
import multiprocessing as mp
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Union

import pandas as pd

from features.add_season_score import add_season_scores
from pruning.prune_outdated_items import prune_outdated_items
from utils.kaggle_tool import KaggleTool
from utils.season import Seasons
from utils.utils import DataFileNames, load_data_from_hnm, get_data_path, load_data, ProjectConfig


def calculate_top_seasonal_sales(df: pd.DataFrame, start_date: datetime.datetime, end_date: datetime.datetime,
                                 top_x: int = 12) -> pd.DataFrame:
    """
    Calculate best-selling seasonal items
    :param df: Dataframe with season scores per item
    :param start_date: Date to start from
    :param end_date: Date to end at
    :param top_x: Calculate top x best-selling seasonal items
    :return: Dataframe with top x best-selling seasonal items for each day
    """
    out_df = pd.DataFrame(columns=['date', 'items'])
    out_df['date'] = pd.to_datetime(out_df['date'])

    for day in pd.date_range(start_date, end_date):
        season = Seasons.get_season(day)
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


def _run_seasonal_analysis(max_score_offset: int, max_score_day_range: int, rerun_seasonal_scores: bool = True,
                           rerun_all: bool = False, to_csv: bool = True, verbose: bool = True,
                           submission_suffix: str = None, do_prune_outdated_items: bool = True) -> Union[Path, BytesIO]:
    # Output paths
    seasonal_scores_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_scores.csv'
    article_sales_per_date_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'article_sales_per_date.csv'
    seasonal_sales_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'seasonal_sales.csv'
    top_seasonal_sales_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'top_seasonal_sales.csv'

    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'), verbose,
                                         dtype={'article_id': str})
    articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'), verbose,
                                     dtype={'article_id': str})

    if do_prune_outdated_items:
        articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df, cutoff_days=365)

    if verbose:
        print(
            f"Using season parameters: max_score_offset={max_score_offset}, max_score_day_range={max_score_day_range}.")

    for season in Seasons.seasons:
        season.set_max_score_offset(max_score_offset)
        season.set_max_score_day_range(max_score_day_range)

    seasonal_scores_df = add_season_scores(transactions_df)

    top_seasonal_sales_df = None
    if not top_seasonal_sales_path.exists() or rerun_seasonal_scores or rerun_all:
        # Calculate top seasonal sales
        top_seasonal_sales_df = calculate_top_seasonal_sales(seasonal_scores_df, ProjectConfig.DATA_END,
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
        output = get_data_path() / DataFileNames.OUTPUT_DIR / (
                'submission' + (f'_{submission_suffix}' if submission_suffix else '') + '.csv')
    else:
        output = BytesIO()

    submission_df.to_csv(output, index=False)

    return output


def already_ran_for(score_offset: int, day_range: int, kaggle_tool: KaggleTool):
    submissions = kaggle_tool.list_submissions_wrapped()

    # Check if submission with same parameters already exists
    for submission in submissions:
        try:
            description = submission.parse_json_description()
        except json.decoder.JSONDecodeError:
            continue
        if description['max_score_offset'] == score_offset and description['max_score_day_range'] == day_range:
            return True

    return False


def run_seasonal_analysis(max_score_offset: int, max_score_day_range: int, check_already_ran: bool = False,
                          rerun_seasonal_scores: bool = True, rerun_all: bool = False,
                          do_prune_outdated_items: bool = True, submit_to_kaggle: bool = True, keep_zip: bool = False):
    """
    Runs seasonal analysis for a given set of parameters.
    :param max_score_offset: Offset from start of season to max score day.
    :param max_score_day_range: Range of days around max score day to calculate score for.
    :param check_already_ran: Whether to check Kaggle submissions if the analysis already ran for the given parameters.
    :param rerun_seasonal_scores: Whether to rerun the seasonal scores.
    :param rerun_all: Whether to rerun all calculations.
    :param do_prune_outdated_items: Whether to prune outdated items.
    :param submit_to_kaggle: Whether to submit the results to Kaggle.
    :param keep_zip: Whether to keep the zip file.
    :return:
    """
    kaggle_tool = KaggleTool('h-and-m-personalized-fashion-recommendations')

    if check_already_ran:
        if already_ran_for(max_score_offset, max_score_day_range, kaggle_tool):
            print(
                f'> Already ran analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')
            return

    print(f'> Running analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')

    start_time = time.time()
    output_bytes = _run_seasonal_analysis(max_score_offset, max_score_day_range,
                                          rerun_seasonal_scores=rerun_seasonal_scores, rerun_all=rerun_all,
                                          verbose=False, to_csv=False, do_prune_outdated_items=do_prune_outdated_items)

    print(
        f'> Finished analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range} in {time.time() - start_time} seconds.')

    if not (submit_to_kaggle or keep_zip):
        return output_bytes

    print(f'> Zipping output')
    zip_path = get_data_path() / DataFileNames.OUTPUT_DIR / DataFileNames.ZIP_DIR / f'seasonal_analysis_{max_score_offset}_{max_score_day_range}.zip'

    if not zip_path.parent.exists():
        zip_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        zip_file.writestr('seasonal_analysis.csv', output_bytes.getvalue())

    if submit_to_kaggle:
        print(f'> Uploading output to Kaggle')
        metadata = {'max_score_offset': max_score_offset, 'max_score_day_range': max_score_day_range,
                    'do_prune_outdated_items': do_prune_outdated_items}

        kaggle_tool.upload_submission(zip_path, metadata=metadata)

    if not keep_zip:
        zip_path.unlink()

    return output_bytes


def run_seasonal_analysis_parallel(_mp_pool_count: int, combinations: list):
    """
    Runs seasonal analysis for a range of parameters and uploads the results to Kaggle.
    Processing is done in parallel using multiprocessing.
    :param _mp_pool_count: Number of processes to use for multiprocessing
    :param combinations: List of tuples containing the parameters to run the analysis for
    """
    with mp.Pool(_mp_pool_count) as pool:
        pool.starmap(_run_seasonal_analysis, combinations)


if __name__ == '__main__':
    _run_seasonal_analysis(-30, 30, True, to_csv=True, verbose=False, submission_suffix='test')
