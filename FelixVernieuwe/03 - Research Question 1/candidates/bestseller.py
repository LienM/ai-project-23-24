import pandas as pd


def candidate_bestsellers_weekly(candidate_customers: pd.DataFrame, most_sold_products_per_week_ranked: pd.DataFrame, k=12):
    """
    For every candidate customer, add the top k products per week as potential candidates
    :param candidate_customers: Customers that can receive candidates across multiple weeks
    :param most_sold_products_per_week_ranked: Top-k best ranked articles per week
    :return: Top-k best ranked articles per week as potential candidates for every customer that bought a product in given period
    """

    # For every customer, add the top-12 best ranked articles as potential candidates for purchases in a given week
    #  using the transaction skeletons
    #       customer_id | channel | week | article_id | bestseller_rank | price
    #       C1234       | 1       | 95   | A5678      | 1               | 0.20
    #       C1234       | 2       | 99   | A1234      | 1               | 0.15
    #       ...

    bestseller_candidates = pd.merge(candidate_customers, most_sold_products_per_week_ranked.groupby('week').head(k), on='week')
    bestseller_candidates.drop(columns=['bestseller_rank'], inplace=True)

    # End result: top k products per week for every candidate customer in given period
    return bestseller_candidates


def candidate_bestsellers_all_time(candidate_customers: pd.DataFrame, most_sold_products_all_time_ranked: pd.DataFrame, k=12):
    """
    For every candidate customer, add the top k products all time as potential candidates
    :param candidate_customers: Customers that can receive candidates across multiple weeks
    :param most_sold_products_all_time_ranked: Top-k best ranked articles all time
    :return: Top-k best ranked articles all time as potential candidates for every customer that bought a product in given period
    """
    bestseller_candidates = pd.merge(candidate_customers, most_sold_products_all_time_ranked.head(k), how='cross')

    return bestseller_candidates


def candidate_bestsellers_age_group(candidate_customers: pd.DataFrame, most_sold_products_per_week_ranked: pd.DataFrame, k=12):

    bestseller_candidates = pd.merge(candidate_customers, most_sold_products_per_week_ranked.groupby('week').head(k), on='week')
    bestseller_candidates.drop(columns=['bestseller_rank'], inplace=True)

    # End result: top k products per week for every candidate customer in given period
    return bestseller_candidates