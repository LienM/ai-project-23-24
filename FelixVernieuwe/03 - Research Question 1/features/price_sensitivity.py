import pandas as pd


def price_sensitivity_feature(transactions: pd.DataFrame, customers: pd.DataFrame):
    """
    Add price sensitivity feature to transactions
    :param transactions: Filtered transactions dataframe
    :param customers: Customers dataframe
    :return: transactions dataframe with price sensitivity feature
    """
    customer_price_ranges = transactions.groupby('customer_id')['price'].agg(['min', 'max'])
    customer_price_ranges['diff'] = customer_price_ranges['max'] - customer_price_ranges['min']
    customer_price_ranges['price_sensitivity'] = pd.qcut(customer_price_ranges['diff'], 5, labels=[0, 1, 2, 3, 4], duplicates='drop')

    customer_price_ranges['purchase_count'] = transactions['customer_id'].value_counts()
    customer_price_ranges.loc[customer_price_ranges['purchase_count'] < 10, 'price_sensitivity'] = 0

    customer_price_ranges.drop(columns=['min', 'max', 'diff', 'purchase_count'], inplace=True)
    customer_price_ranges.reset_index(inplace=True)

    customers = customers.merge(customer_price_ranges, on='customer_id', how='left').fillna(0)
    transactions = transactions.merge(customers[['customer_id', 'price_sensitivity']], on='customer_id', how='left')

    return transactions