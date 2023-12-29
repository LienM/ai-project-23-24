import pandas as pd


def candidate_bestsellers_weekly(candidate_customers: pd.DataFrame, bestsellers_weekly: pd.DataFrame, k=12):
    """
    For every candidate customer, add the top k products per week as potential candidates
    :param candidate_customers: Customers that can receive candidates across multiple weeks
    :param bestsellers_weekly: Top-k best ranked articles per week
    :param k: Amount of top articles per week to add as candidates
    :return: Top-k best ranked articles per week as potential candidates for every customer that bought a product in given period
    """

    # For every customer, add the top-12 best ranked articles as potential candidates for purchases in a given week
    #  using the transaction skeletons
    #       customer_id | channel | week | article_id | bestseller_rank | price
    #       C1234       | 1       | 95   | A5678      | 1               | 0.20
    #       C1234       | 2       | 99   | A1234      | 1               | 0.15
    #       ...

    # For every week, get the top 12 most sold products
    bestsellers_weekly = bestsellers_weekly.sort_values(['week', 'rank']).groupby('week').head(k)
    bestsellers_weekly = bestsellers_weekly.copy()

    # NOTE: since reference week is not in our bestseller dataset, we need to do one of two following actions:
    #   1. Increment all weeks by one (Radek's solution and also current one)
    #   2. For reference week, always select the last week's bestsellers
    # -> This is implemented in the bestseller generation proces
    bestsellers_weekly.drop(columns=['rank', 'count'], inplace=True)

    # Cross merge
    bestseller_candidates = pd.merge(candidate_customers, bestsellers_weekly, on='week')

    # End result: top k products per week for every candidate customer in given period

    return bestseller_candidates


def candidate_bestsellers_all_time(candidate_customers: pd.DataFrame, bestsellers_all_time: pd.DataFrame, k=12):
    """
    For every candidate customer, add the top k products all time as potential candidates
    :param candidate_customers: Customers that can receive candidates across multiple weeks
    :param bestsellers_all_time: Top-k best ranked articles all time
    :param k: Amount of top articles to add as candidates
    :return: Top-k best ranked articles all time as potential candidates for every customer that bought a product in given period
    """
    bestsellers_all_time.drop(columns=['rank', 'count'], inplace=True)
    bestseller_candidates = pd.merge(candidate_customers, bestsellers_all_time.head(k), how='cross')

    return bestseller_candidates


def candidate_bestsellers_age_group(candidate_customers: pd.DataFrame, customers: pd.DataFrame, bestsellers_age_group: pd.DataFrame, k=12):
    """
    For every candidate customer, add the top k products per age group (for that week) as potential candidates
    :param candidate_customers: Customers that can receive candidates across multiple weeks
    :param transactions: Transactions dataset (must contain weeks used by candidate customers)
    :param customers: Customers dataset
    :param k: Amount of top articles per age group to add as candidates
    :return: Top-k best ranked articles per age group per week as potential candidates for every customer that bought a product in given period
    """

    # Re-assign age group to candidate customers
    candidate_customers = pd.merge(candidate_customers, customers[['customer_id', 'age']], on='customer_id')

    # For every age_group get the top k best ranked articles per week
    bestsellers_age_group = bestsellers_age_group.groupby(['week', 'age']).head(k)

    # Merge the top k articles per age group with the candidate customers
    bestseller_candidates = pd.merge(candidate_customers, bestsellers_age_group, on=['week', 'age'])
    bestseller_candidates.drop(columns=['age'], inplace=True)

    return bestseller_candidates
