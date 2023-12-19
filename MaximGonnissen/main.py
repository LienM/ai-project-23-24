import multiprocessing as mp
import time

from analysis.seasonality_analysis import run_seasonal_analysis_parallel, run_seasonal_analysis
from utils.convert_to_parquet import convert_to_parquet
from utils.utils import DataFileNames, get_data_path


def get_seasonality_combinations():
    max_score_offset_range = range(-60, 10, 1)
    max_score_day_range_range = range(7, 30, 1)

    combinations = [(max_score_offset, max_score_day_range) for max_score_offset in max_score_offset_range for
                    max_score_day_range in max_score_day_range_range]
    return combinations


def seasonal_analysis():
    use_mp = False

    if use_mp:
        mp_pool_count = max(mp.cpu_count() - 1, 1)
        print(f'Using {mp_pool_count} cores for multiprocessing.')

        run_seasonal_analysis_parallel(mp_pool_count, get_seasonality_combinations())

    else:
        for combination in get_seasonality_combinations():
            run_seasonal_analysis(*combination)


def convert_all_to_parquet():
    data_path = get_data_path() / DataFileNames.HNM_DIR

    convert_to_parquet(data_path / DataFileNames.CUSTOMERS)
    convert_to_parquet(data_path / DataFileNames.ARTICLES)
    convert_to_parquet(data_path / DataFileNames.TRANSACTIONS_TRAIN)
    convert_to_parquet(data_path / DataFileNames.SAMPLE_SUBMISSION)


if __name__ == '__main__':
    print('Starting script at', time.strftime("%H:%M:%S", time.localtime()))

    # Plot pruning
    run_seasonal_analysis(-30, 60, rerun_all=True, do_prune_outdated_items=True)

    print('Finished script at', time.strftime("%H:%M:%S", time.localtime()))
