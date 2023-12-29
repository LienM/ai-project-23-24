import pandas as pd


def remove_active_nulls(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    @DEPRECATED: This function is deprecated because it removes too many customers.

    Remove customers with a null value in the 'active' column from the customers.csv DataFrame.

    :param customers_df: DataFrame containing customer information
    :return: DataFrame containing only active customers
    """
    raise DeprecationWarning('This function is deprecated because it removes too many customers.')
    return customers_df[customers_df['Active'].notnull()]


def remove_active_pre_create(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove customers with a 'pre_create' value in the 'active' column from the customers.csv DataFrame.

    :param customers_df: DataFrame containing customer information
    :return: DataFrame containing only active customers
    """
    return customers_df[customers_df['Active'] != 'PRE-CREATE']


def prune_inactive(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prune inactive customers from the customers.csv DataFrame.
    Inactive customers are defined as customers who appear to not have finished creating their account.

    :param customers_df: DataFrame containing customer information
    :return: DataFrame containing only active customers
    """
    # customers_df = remove_active_nulls(customers_df)  # --> Removes significant amount of customers
    customers_df = remove_active_pre_create(customers_df)

    return customers_df
