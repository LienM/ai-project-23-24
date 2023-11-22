import multiprocessing as mp

from analysis.seasonality_analysis import run_seasonal_analysis_parallel, run_seasonal_analysis
from features.add_gender import add_gender
from pruning.prune_inactive import prune_inactive
from pruning.prune_no_purchases import prune_no_purchases
from utils.utils import load_data_from_hnm, DataFileNames


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


def get_gendered_customers_df():
    customers_df = load_data_from_hnm(DataFileNames.CUSTOMERS)
    customers_df = prune_inactive(customers_df)

    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN)
    customers_df = prune_no_purchases(customers_df, transactions_df)

    articles_df = load_data_from_hnm(DataFileNames.ARTICLES)

    return add_gender(customers_df, transactions_df, articles_df)


if __name__ == '__main__':
    print(get_gendered_customers_df().head(100))
