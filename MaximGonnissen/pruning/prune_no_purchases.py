import pandas as pd


def prune_no_purchases(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prune customers who have not made any purchases from the customers.csv DataFrame.

    :param customers_df: DataFrame containing customer information
    :param transactions_df: DataFrame containing transaction information
    :return: DataFrame containing only active customers
    """
    customers_df = customers_df[customers_df['customer_id'].isin(transactions_df['customer_id'])]

    return customers_df
