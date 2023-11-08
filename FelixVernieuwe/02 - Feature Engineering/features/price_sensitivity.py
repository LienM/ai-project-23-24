import pandas as pd


def price_sensitivity_feature(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame):
    customer_price_ranges = transactions.groupby('customer_id')['price'].agg(['min', 'max'])
    customer_price_ranges['diff'] = customer_price_ranges['max'] - customer_price_ranges['min']
    customer_price_ranges['price_sensitivity'] = pd.qcut(customer_price_ranges['diff'], 5, labels=[0, 1, 2, 3, 4])

    customer_price_ranges['purchase_count'] = transactions['customer_id'].value_counts()
    customer_price_ranges.loc[customer_price_ranges['purchase_count'] < 10, 'price_sensitivity'] = 0

    customer_price_ranges.drop(columns=['min', 'max', 'diff', 'purchase_count'], inplace=True)

    customers = customers.merge(customer_price_ranges, on='customer_id', how='left').fillna(0)

    return transactions, customers, articles