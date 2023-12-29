from pathlib import Path

import pandas as pd

from utils.utils import DataFileNames, get_data_path


def convert_to_parquet(path: Path) -> None:
    """
    Convert a dataframe to parquet format and save it to the given path.
    :param path: Path to save the dataframe to.
    """
    dataframe = pd.read_csv(path, dtype={'article_id': str})
    if 't_dat' in dataframe.columns:
        dataframe['t_dat'] = pd.to_datetime(dataframe['t_dat'])
    dataframe.to_parquet(path.with_suffix('.parquet'))


def convert_all_to_parquet():
    data_path = get_data_path() / DataFileNames.HNM_DIR

    convert_to_parquet(data_path / DataFileNames.CUSTOMERS)
    convert_to_parquet(data_path / DataFileNames.ARTICLES)
    convert_to_parquet(data_path / DataFileNames.TRANSACTIONS_TRAIN)
    convert_to_parquet(data_path / DataFileNames.SAMPLE_SUBMISSION)
