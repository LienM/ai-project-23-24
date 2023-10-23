import time
import multiprocessing as mp
from utils import DataFileNames, load_data_from_hnm, get_data_path

import pandas as pd


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


if __name__ == '__main__':
    script_start_time = time.time()
    mp_pool_count = max(mp.cpu_count() - 1, 1)
    print(f'Using {mp_pool_count} cores for multiprocessing.')

    # Load data
    transactions_train = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN)

    article_sales_per_date_df = article_sales_per_date(transactions_train)

    output_path = get_data_path() / DataFileNames.OUTPUT_DIR / 'article_sales_per_date.csv'

    article_sales_per_date_df.to_csv(output_path, index=False)
