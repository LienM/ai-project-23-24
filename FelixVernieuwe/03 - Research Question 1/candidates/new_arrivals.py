import pandas as pd

from data.candidate_products import get_first_sale_products


def candidate_new_arrivals(candidate_customers: pd.DataFrame, all_transactions: pd.DataFrame, k=5, max_age=0):
    """
    For every customer candidate, add the new arrivals of the week as potential candidates
    :param candidate_customers: Customers that can receive candidates for every week in given period
    :param all_transactions: Transactions in the entire dataset (used for finding relative age of product)
    :param k: Number of new arrivals to add as candidates
    :param max_age: Maximum age of new arrivals to add as candidates
    :return: New arrivals as potential candidates for every customer that bought a product in given period
    """

    first_sale = get_first_sale_products(all_transactions)

    all_transactions['first_week'] = all_transactions['article_id'].map(first_sale.set_index('article_id')['first_week'])
    all_transactions['prod_age'] = all_transactions['week'] - all_transactions['first_week']
    all_transactions.drop(columns=['first_week'], inplace=True)

    # New arrivals are all product whose age is equal to or below the max_age in the data
    new_arrivals = all_transactions[all_transactions['prod_age'] <= max_age]

    # Get the total amount of sales for new_arrival product (across multiple weeks), then sort by week
    total_sales_period = new_arrivals.groupby(['article_id']).size().reset_index(name='count')[['article_id', 'count']]

    new_arrivals = pd.merge(new_arrivals, total_sales_period, on='article_id', how='left')

    # Group by article_id and week, do not keep count order by count
    new_arrivals = new_arrivals.groupby(['week', 'article_id']).head(1)

    # Order by week and then sort by count
    new_arrivals = new_arrivals.sort_values(by=["week", "count"], ascending=[True, False])[["week", "article_id", "count"]]
    new_arrivals = new_arrivals.groupby(['week']).head(k)

    new_arrivals_candidates = pd.merge(candidate_customers, new_arrivals, on='week', how="cross")

    return new_arrivals_candidates
