import pathlib
import time
from datetime import datetime
from typing import Tuple

import pandas as pd


class ProjectConfig:
    """
    Static class to hold important variables
    """
    DATA_START = datetime.fromisoformat("2018-09-20")
    DATA_END = datetime.fromisoformat("2020-09-22")


class DataFileNames:
    """
    Static class containing the names of the data files and directories.
    """
    DATA_DIR = 'data'
    HNM_DIR = 'h-and-m-personalized-fashion-recommendations'
    ARTICLES = 'articles.csv'
    CUSTOMERS = 'customers.csv'
    SAMPLE_SUBMISSION = 'sample_submission.csv'
    TRANSACTIONS_TRAIN = 'transactions_train.csv'
    IMAGES_DIR = 'images'
    OUTPUT_DIR = 'output'
    PLOTS_DIR = 'plots'
    FEATURE_ENGINEERING_DIR = 'feature_engineering'
    ZIP_DIR = 'zips'

    @staticmethod
    def as_parquet(path: pathlib.Path) -> pathlib.Path:
        """
        Get the path to the parquet file.
        :param path: Path to the csv file.
        :return: Path to the parquet file.
        """
        return path.with_suffix('.parquet')

    @staticmethod
    def as_csv(path: pathlib.Path) -> pathlib.Path:
        """
        Get the path to the csv file.
        :param path: Path to the parquet file.
        :return: Path to the csv file.
        """
        return path.with_suffix('.csv')


def get_data_path() -> pathlib.Path:
    """
    Get the path to the data folder.
    Assumes the data folder is in the same folder as the script, or in the parent folder.
    :return: Path to the data folder.
    """
    data_path = pathlib.Path.cwd() / DataFileNames.DATA_DIR
    if not data_path.exists():
        data_path = pathlib.Path.cwd().parent / DataFileNames.DATA_DIR
    if not data_path.exists():
        raise FileNotFoundError('Could not find data folder.')
    return data_path


def load_data(path: pathlib.Path, verbose: bool = True, as_parquet: bool = False, **kwargs) -> pd.DataFrame:
    """
    Load data from a csv file, measuring the time it takes to load.
    :param path: Path to the csv file.
    :param verbose: Whether to print information about the loading process.
    :param as_parquet: Whether to load the data from a parquet file instead of a csv file.
    :return: Dataframe containing the data from the csv file.
    """
    time_start = time.time()
    if verbose:
        print(f'[ ] Loading data from {path}...')
    if as_parquet:
        path = DataFileNames.as_parquet(path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, **kwargs)
    if verbose:
        print(f'[X] Loaded data from {path} in {time.time() - time_start:.2f} seconds.')
    return df


def load_data_from_hnm(path: pathlib.Path, verbose: bool = True, as_parquet: bool = False, **kwargs) -> pd.DataFrame:
    """
    Load data from a csv file in the h-and-m-personalized-fashion-recommendations folder,
    measuring the time it takes to load.
    :param path: Path to the csv file.
    :param verbose: Whether to print information about the loading process.
    :param as_parquet: Whether to load the data from a parquet file instead of a csv file.
    :return: Dataframe containing the data from the csv file.
    """
    return load_data(get_data_path() / DataFileNames.HNM_DIR / path, verbose, **kwargs)


def get_all_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns all DataFrames.
    :return: (articles_df, customers_df, transactions_df, submissions_df)
    """
    articles_df = load_data_from_hnm(DataFileNames.ARTICLES, as_parquet=True)
    customers_df = load_data_from_hnm(DataFileNames.CUSTOMERS, as_parquet=True)
    transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN, as_parquet=True)
    submissions_df = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION, as_parquet=True)

    return articles_df, customers_df, transactions_df, submissions_df
