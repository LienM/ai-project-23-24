import pandas as pd
from pathlib import Path


def convert_to_parquet(path: Path) -> None:
    """
    Convert a dataframe to parquet format and save it to the given path.
    :param path: Path to save the dataframe to.
    """
    dataframe = pd.read_csv(path)
    dataframe.to_parquet(str(path).replace('.csv', '.parquet'))
