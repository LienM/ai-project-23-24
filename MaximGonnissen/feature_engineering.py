from utils import get_data_path, load_data, DataFileNames, load_data_from_hnm
from progress_bar import ProgressBar, ASCIIColour
import multiprocessing as mp
import time

import pandas as pd


class AgeGroup:
    """
    Static class to hold age group constants.
    """
    YOUNG = '0-20'
    YOUNG_ADULT = '20-30'
    ADULT = '30-40'
    MIDDLE_AGED = '40-60'
    OLD = '60+'


def add_month_year_features(df: pd.DataFrame):
    """
    Adds month and year features to the dataframe, using the t_dat column as origin.
    :param df: Transactions dataframe to add features to.
    :return: Transactions dataframe with added features.
    """
    start_time = time.time()
    print('[ ] Adding month and year features to transactions dataframe...')
    new_df = df.copy()
    new_df['t_dat'] = pd.to_datetime(new_df['t_dat'])
    new_df['month'] = new_df['t_dat'].dt.month
    new_df['year'] = new_df['t_dat'].dt.year
    print(f'[X] Added month and year features to transactions dataframe in {time.time() - start_time:.2f} seconds.')
    return new_df


def add_age_group_feature(df: pd.DataFrame):
    """
    Adds age group features to the dataframe, using the age column as origin.
    :param df: Customers dataframe to add features to.
    :return: Customers dataframe with added features.
    """
    start_time = time.time()
    print('[ ] Adding age group features to customers dataframe...')
    new_df = df.copy()
    new_df['age_group'] = pd.cut(new_df['age'], bins=[0, 20, 30, 40, 60, 100],
                                 labels=[AgeGroup.YOUNG, AgeGroup.YOUNG_ADULT, AgeGroup.ADULT, AgeGroup.MIDDLE_AGED,
                                         AgeGroup.OLD])
    print(f'[X] Added age group features to customers dataframe in {time.time() - start_time:.2f} seconds.')
    return new_df


def get_purchases_for_customer(customer_id: int, transactions: pd.DataFrame):
    """
    Get all purchases for a customer.
    :param customer_id: ID of the customer to get purchases for.
    :param transactions: Transactions dataframe to get purchases from.
    :return: Dataframe containing all purchases for the customer.
    """
    return transactions[transactions['customer_id'] == customer_id]


def get_colour_group_name_for_article(article_id: int, articles_: pd.DataFrame):
    """
    Get the colour group name for an article.
    :param article_id: ID of the article to get the colour group name for.
    :param articles_: Articles dataframe to get the colour group name from.
    :return: Colour group name for the article.
    """
    return articles_[articles_['article_id'] == article_id]['colour_group_name'].iloc[0]


def get_index_group_name_for_article(article_id: int, articles_: pd.DataFrame):
    """
    Get the index group name for an article.
    :param article_id: ID of the article to get the index group name for.
    :param articles_: Articles dataframe to get the index group name from.
    :return: Index group name for the article.
    """
    return articles_[articles_['article_id'] == article_id]['index_group_name'].iloc[0]


def add_favourite_colour_feature(customers_: pd.DataFrame, articles_: pd.DataFrame, transactions_: pd.DataFrame):
    """
    Adds a favourite colour feature to the customers dataframe.
    :param customers_: Customers dataframe to add the feature to.
    :param articles_: Articles dataframe to get the colour group names from.
    :param transactions_: Transactions dataframe to get the purchases from.
    :return: Customers dataframe with added feature.
    """
    start_time = time.time()
    print('[ ] Adding favourite colour feature to customers dataframe...')
    new_customers_ = customers_.copy()
    new_customers_['favourite_colour'] = None
    customer_ids = customers_['customer_id'].unique()
    with ProgressBar(customer_ids, ansi_colour=ASCIIColour.GREEN) as customer_ids_bar:
        for customer_id in customer_ids_bar:
            purchases = get_purchases_for_customer(customer_id, transactions_)
            if purchases.empty:
                continue
            favourite_colour = purchases['article_id'].apply(
                lambda article_id: get_colour_group_name_for_article(article_id, articles_)).value_counts().index[0]
            new_customers_.loc[new_customers_['customer_id'] == customer_id, 'favourite_colour'] = favourite_colour
    print(f'[X] Added favourite colour feature to customers dataframe in {time.time() - start_time:.2f} seconds.')
    return new_customers_


def add_favourite_index_feature(customers_: pd.DataFrame, articles_: pd.DataFrame, transactions_: pd.DataFrame):
    """
    Adds a favourite index feature to the customers dataframe.
    :param customers_: Customers dataframe to add the feature to.
    :param articles_: Articles dataframe to get the index group names from.
    :param transactions_: Transactions dataframe to get the purchases from.
    :return: Customers dataframe with added feature.
    """
    start_time = time.time()
    print('[ ] Adding favourite index feature to customers dataframe...')
    new_customers_ = customers_.copy()
    new_customers_['favourite_index'] = None
    customer_ids = customers_['customer_id'].unique()
    with ProgressBar(customer_ids, ansi_colour=ASCIIColour.GREEN) as customer_ids_bar:
        for customer_id in customer_ids_bar:
            purchases = get_purchases_for_customer(customer_id, transactions_)
            if purchases.empty:
                continue
            favourite_index = purchases['article_id'].apply(
                lambda article_id: get_index_group_name_for_article(article_id, articles_)).value_counts().index[0]
            new_customers_.loc[new_customers_['customer_id'] == customer_id, 'favourite_index'] = favourite_index
    print(f'[X] Added favourite index feature to customers dataframe in {time.time() - start_time:.2f} seconds.')
    return new_customers_


if __name__ == '__main__':
    script_start_time = time.time()
    # Calculate max number of processes to allow for multiprocessing
    mp_pool_count = max(mp.cpu_count() - 1, 1)
    print(f'Using {mp_pool_count} cores for multiprocessing.')

    data_path = get_data_path()

    articles = load_data_from_hnm(DataFileNames.ARTICLES)
    customers = load_data_from_hnm(DataFileNames.CUSTOMERS)
    sample_submission = load_data_from_hnm(DataFileNames.SAMPLE_SUBMISSION)
    transactions_train = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN)

    export_path = data_path / DataFileNames.OUTPUT_DIR / DataFileNames.FEATURE_ENGINEERING_DIR

    if not export_path.exists():
        export_path.mkdir(parents=True)

    new_transactions_train = add_month_year_features(transactions_train)
    save_time_start = time.time()
    new_transactions_train.to_csv(export_path / DataFileNames.TRANSACTIONS_TRAIN, index=False)
    print(f'[X] Saved transactions_train in {time.time() - save_time_start:.2f} seconds.')

    new_customers = add_age_group_feature(customers)
    save_time_start = time.time()
    new_customers.to_csv(export_path / DataFileNames.CUSTOMERS, index=False)
    print(f'[X] Saved customers in {time.time() - save_time_start:.2f} seconds.')

    new_customers = add_favourite_colour_feature(new_customers, articles, transactions_train)
    save_time_start = time.time()
    new_customers.to_csv(export_path / DataFileNames.CUSTOMERS, index=False)
    print(f'[X] Saved customers in {time.time() - save_time_start:.2f} seconds.')

    new_customers = add_favourite_index_feature(new_customers, articles, transactions_train)
    save_time_start = time.time()
    new_customers.to_csv(export_path / DataFileNames.CUSTOMERS, index=False)
    print(f'[X] Saved customers in {time.time() - save_time_start:.2f} seconds.')