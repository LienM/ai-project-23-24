import pathlib
import time
from datetime import datetime

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


def load_data(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    """
    Load data from a csv file, measuring the time it takes to load.
    :param path: Path to the csv file.
    :return: Dataframe containing the data from the csv file.
    """
    time_start = time.time()
    print(f'[ ] Loading data from {path}...')
    df = pd.read_csv(path, **kwargs)
    print(f'[X] Loaded data from {path} in {time.time() - time_start:.2f} seconds.')
    return df


def load_data_from_hnm(path: pathlib.Path, **kwargs) -> pd.DataFrame:
    """
    Load data from a csv file in the h-and-m-personalized-fashion-recommendations folder,
    measuring the time it takes to load.
    :param path: Path to the csv file.
    :return: Dataframe containing the data from the csv file.
    """
    return load_data(get_data_path() / DataFileNames.HNM_DIR / path, **kwargs)
