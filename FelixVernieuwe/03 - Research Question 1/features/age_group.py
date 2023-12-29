import pandas as pd


def age_group_feature(transactions: pd.DataFrame, customers: pd.DataFrame):
    """
    Add age group of the customer buying the product
    :param transactions: Filtered transactions dataframe
    :param customers: Customers dataframe
    :returns: transactions dataframe with age group feature
    """

    # Cut ages into bins
    transactions = pd.merge(transactions, customers[['customer_id', 'age']], on='customer_id')

    transactions["age"] = pd.cut(transactions['age'], bins=[-1, 0, 21, 30, 45, 70], labels=False)

    return transactions
