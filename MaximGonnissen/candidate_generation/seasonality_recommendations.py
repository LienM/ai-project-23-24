import datetime
import json
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Union, Optional

import pandas as pd

from features.add_season_score import add_season_scores
from pruning.prune_outdated_items import prune_outdated_items
from utils.kaggle_tool import KaggleTool
from utils.season import Seasons
from utils.utils import DataFileNames, load_data_from_hnm, get_data_path, ProjectConfig


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


def predict_top_seasonal_items(seasonal_sales_df: pd.DataFrame) -> list:
    """
    Predict most popular items based on top seasonal sales
    :param seasonal_sales_df: Dataframe with top items per day based on season scores, for prediction days
    :return: List of top items based on top seasonal sales
    """
    item_scores = {}
    for index, row in seasonal_sales_df.iterrows():
        items = row['items'].split(' ')
        for i in range(len(items)):
            item_scores[items[i]] = item_scores.get(items[i], 0) + len(items) - i

    return sorted(item_scores, key=item_scores.get, reverse=True)[:len(seasonal_sales_df['items'].iloc[0].split(' '))]


def _run_seasonal_recommendations(max_score_offset: int, max_score_day_range: int, to_csv: bool = True,
                                  verbose: bool = True, submission_suffix: str = None,
                                  do_prune_outdated_items: bool = True) -> Union[Path, BytesIO]:
    """
    Runs seasonal recommendations for a given set of parameters.
    :param max_score_offset: Offset from start of season to max score day. (Note that this should be negative to be *before* the season starts)
    :param max_score_day_range: Range of days around max score day to calculate score for.
    :param to_csv: Whether to save the submission to a csv file.
    :param verbose: Whether to run all calculations verbose.
    :param submission_suffix: Suffix to add to the submission file name.
    :param do_prune_outdated_items: Whether to prune outdated items.
    :return: Path to the submission csv file if to_csv is True, else the BytesIO object containing the csv file.
    """
    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'), verbose,
                                         dtype={'article_id': str})
    articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'), verbose,
                                     dtype={'article_id': str})

    if do_prune_outdated_items:
        articles_df, transactions_df = prune_outdated_items(articles_df, transactions_df, cutoff_days=365)

    if verbose:
        print(f"Using parameters: max_score_offset={max_score_offset}, max_score_day_range={max_score_day_range}.")

    for season in Seasons.seasons:
        season.set_max_score_offset(max_score_offset)
        season.set_max_score_day_range(max_score_day_range)

    seasonal_scores_df = add_season_scores(transactions_df)

    # Calculate top seasonal sales
    top_seasonal_sales_df = calculate_top_seasonal_sales(seasonal_scores_df, ProjectConfig.DATA_END,
                                                         ProjectConfig.DATA_END + datetime.timedelta(days=7))

    # Predict top seasonal items
    top_items = predict_top_seasonal_items(top_seasonal_sales_df)
    top_items_string = ' '.join([str(item_id) for item_id in top_items])

    # Create submission
    submission_df = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION, verbose)
    submission_df['prediction'] = top_items_string

    if to_csv:
        output = get_data_path() / DataFileNames.OUTPUT_DIR / (
                'submission' + (f'_{submission_suffix}' if submission_suffix else '') + '.csv')
    else:
        output = BytesIO()

    submission_df.to_csv(output, index=False)

    return output


def already_ran_for(score_offset: int, day_range: int, kaggle_tool: KaggleTool) -> bool:
    """
    Checks if the recommendations already ran for the given parameters.
    :param score_offset: Offset from start of season to max score day.
    :param day_range: Range of days around max score day to calculate score for.
    :param kaggle_tool: KaggleTool object to use for checking submissions.
    :return: Whether the recommendations already ran for the given parameters.
    """
    submissions = kaggle_tool.list_submissions_wrapped()

    for submission in submissions:
        try:
            description = submission.parse_json_description()
        except json.decoder.JSONDecodeError:
            continue
        if description['max_score_offset'] == score_offset and description['max_score_day_range'] == day_range:
            return True

    return False


def run_seasonal_recommendations(max_score_offset: int, max_score_day_range: int, check_already_ran: bool = False,
                                 do_prune_outdated_items: bool = True, submit_to_kaggle: bool = True,
                                 keep_zip: bool = False) -> Optional[BytesIO]:
    """
    Runs seasonal recommendations for a given set of parameters. Optionally uploads the results to Kaggle.
    :param max_score_offset: Offset from start of season to max score day.
    :param max_score_day_range: Range of days around max score day to calculate score for.
    :param check_already_ran: Whether to check Kaggle submissions if the recommendations already ran for the given parameters.
    :param do_prune_outdated_items: Whether to prune outdated items.
    :param submit_to_kaggle: Whether to submit the results to Kaggle.
    :param keep_zip: Whether to keep the zip file.
    :return:
    """
    kaggle_tool = KaggleTool('h-and-m-personalized-fashion-recommendations')

    if check_already_ran:
        if already_ran_for(max_score_offset, max_score_day_range, kaggle_tool):
            print(
                f'> Already ran recommendations for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')
            return None

    print(
        f'> Running recommendations for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')

    start_time = time.time()
    output_bytes = _run_seasonal_recommendations(max_score_offset, max_score_day_range, verbose=False, to_csv=False,
                                                 do_prune_outdated_items=do_prune_outdated_items)

    print(
        f'> Finished recommendations for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range} in {time.time() - start_time} seconds.')

    if not (submit_to_kaggle or keep_zip):
        return output_bytes

    print(f'> Zipping output')
    zip_path = get_data_path() / DataFileNames.OUTPUT_DIR / DataFileNames.ZIP_DIR / f'seasonal_recommendations_{max_score_offset}_{max_score_day_range}.zip'

    if not zip_path.parent.exists():
        zip_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        zip_file.writestr('seasonal_recommendations.csv', output_bytes.getvalue())

    if submit_to_kaggle:
        print(f'> Uploading output to Kaggle')
        metadata = {'max_score_offset': max_score_offset, 'max_score_day_range': max_score_day_range,
                    'do_prune_outdated_items': do_prune_outdated_items}

        kaggle_tool.upload_submission(zip_path, metadata=metadata)

    if not keep_zip:
        zip_path.unlink()

    return output_bytes
