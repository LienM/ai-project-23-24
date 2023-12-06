import pandas as pd
from pathlib import Path
from datetime import datetime


def convert_to_parquet(path: Path) -> None:
    """
    Convert a dataframe to parquet format and save it to the given path.
    :param path: Path to save the dataframe to.
    """
    dataframe = pd.read_csv(path, dtype={'article_id': str})
    if 't_dat' in dataframe.columns:
        dataframe['t_dat'] = pd.to_datetime(dataframe['t_dat'])
    dataframe.to_parquet(path.with_suffix('.parquet'))
