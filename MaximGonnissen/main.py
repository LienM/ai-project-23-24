from utils.kaggle_tool import KaggleTool
from analysis.seasonality_analysis import run_seasonal_analysis
from utils.utils import DataFileNames, get_data_path
import time
import zipfile
import multiprocessing as mp
import json


def already_ran_for(score_offset: int, day_range: int, kaggle_tool: KaggleTool):
    submissions = kaggle_tool.list_submissions_wrapped()

    print([submission.get_filename() for submission in submissions])

    # Check if submission with same parameters already exists
    for submission in submissions:
        try:
            description = submission.parse_json_description()
        except json.decoder.JSONDecodeError:
            continue
        if description['max_score_offset'] == score_offset and description['max_score_day_range'] == day_range:
            return True

    return False


def get_combinations():
    max_score_offset_range = range(-60, 10, 1)
    max_score_day_range_range = range(7, 30, 1)

    combinations = [(max_score_offset, max_score_day_range) for max_score_offset in max_score_offset_range for max_score_day_range in max_score_day_range_range]
    return combinations


def _run_seasonal_analysis(max_score_offset: int, max_score_day_range: int):
    kaggle_tool = KaggleTool('h-and-m-personalized-fashion-recommendations')

    if already_ran_for(max_score_offset, max_score_day_range, kaggle_tool):
        print(
            f'> Already ran analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')
        return

    print(
        f'> Running analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range}.')

    start_time = time.time()
    output_bytes = run_seasonal_analysis(max_score_offset, max_score_day_range, True, verbose=False, to_csv=False)

    print(
        f'> Finished analysis for max_score_offset={max_score_offset} and max_score_day_range={max_score_day_range} in {time.time() - start_time} seconds.')

    print(f'> Zipping output')
    zip_path = get_data_path() / DataFileNames.OUTPUT_DIR / DataFileNames.ZIP_DIR / f'seasonal_analysis_{max_score_offset}_{max_score_day_range}.zip'

    if not zip_path.parent.exists():
        zip_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        zip_file.writestr('seasonal_analysis.csv', output_bytes.getvalue())

    print(f'> Uploading output to Kaggle')
    metadata = {'max_score_offset': max_score_offset, 'max_score_day_range': max_score_day_range, }
    kaggle_tool.upload_submission(zip_path, metadata=metadata)

    return


def run_seasonal_analysis_parallel(_mp_pool_count: int):
    """
    Runs seasonal analysis for a range of parameters and uploads the results to Kaggle.
    Processing is done in parallel using multiprocessing.
    :param _mp_pool_count: Number of processes to use for multiprocessing
    """
    with mp.Pool(_mp_pool_count) as pool:
        pool.starmap(_run_seasonal_analysis, get_combinations())


if __name__ == '__main__':
    script_start_time = time.time()
    use_mp = False

    if use_mp:
        mp_pool_count = max(mp.cpu_count() - 1, 1)
        print(f'Using {mp_pool_count} cores for multiprocessing.')

        run_seasonal_analysis_parallel(mp_pool_count)

    else:
        for combination in get_combinations():
            _run_seasonal_analysis(*combination)
