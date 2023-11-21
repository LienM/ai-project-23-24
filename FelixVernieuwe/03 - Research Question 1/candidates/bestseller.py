import pandas as pd


def candidate_bestsellers_weekly(candidate_customers: pd.DataFrame, ref_week_candidate_customers: pd.DataFrame,
                            most_sold_products_per_week_ranked: pd.DataFrame):
    """
    For every candidate customer, add the top 12 products per week as potential candidates
    :param candidate_customers: Customers that can receive candidates for every week in the training period
    :param ref_week_candidate_customers: Customers that can receive candidates in the reference week
    :param most_sold_products_per_week_ranked: Top-12 best ranked articles per week
    :return: Top-12 best ranked articles per week as potential candidates for every customer that bought a product in that week and the reference week
    """

    # For every customer, add the top-12 best ranked articles as potential candidates for purchases in a given week
    #  using the transaction skeletons
    #       customer_id | channel | week | article_id | bestseller_rank | price
    #       C1234       | 1       | 95   | A5678      | 1               | 0.20
    #       C1234       | 2       | 99   | A1234      | 1               | 0.15
    #       ...
    candidates_training_interval = pd.merge(candidate_customers, most_sold_products_per_week_ranked, on='week')

    # For every customer, add the top-12 best ranked articles as potential candidates for purchases in reference week
    #       customer_id | channel | week | article_id | bestseller_rank | price
    #       C1234       | 1       | 104  | A5678      | 1               | 0.20
    #       ...
    candidates_reference_week = pd.merge(ref_week_candidate_customers, most_sold_products_per_week_ranked, on='week')

    # Combine all candidates from training period and reference week
    #       customer_id | channel | week | article_id | bestseller_rank | price
    #       C1234       | 1       | 95   | A5678      | 1               | 0.20
    #       C1234       | 2       | 99   | A1234      | 1               | 0.15
    #       C1234       | 1       | 104  | A5678      | 1               | 0.20
    all_candidate = pd.concat([candidates_training_interval, candidates_reference_week])
    all_candidate.drop(columns=['bestseller_rank'], inplace=True)

    # End result: top 12 products per week for every candidate customer in training period and reference week
    return all_candidate
