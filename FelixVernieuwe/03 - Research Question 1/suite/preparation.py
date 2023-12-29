import sys
import logging
import pandas as pd

sys.path.append("..")

from candidates import generate_candidates, generate_candidates_single
from features import weekly_bestseller_feature, all_time_bestseller_feature, price_sensitivity_feature, \
                        product_age_feature, discount_feature, age_group_feature
from data import get_buying_customers_candidates, get_most_sold_products, get_most_sold_weekly_products, \
                    get_most_sold_weekly_age_group_products, get_first_sale_products

from util import split_dataframe, join_all_features


# Re-Export generate_missing_candidates from the candidates
def generate_missing_candidates(method, all_transactions, transactions, customers, reference_week, predicted_candidates):
    """
    Generates missing candidates for the customers that are not in the predicted candidates set
    :param method: Candidate generation method to use
    :param all_transactions: Full transactions dataframe
    :param transactions: Filtered transactions dataframe
    :param customers: Customers dataframe
    :param reference_week: Week to generate candidates for
    :param predicted_candidates: Predicted candidates dataframe
    :return: Candidates for missing customers
    """
    existing_customers = predicted_candidates["customer_id"].unique()
    missing_customers = customers[~customers["customer_id"].isin(existing_customers)]
    logging.debug(f"[CANDIDATES] Found {len(missing_customers)} missing customers, out of {len(customers)} total customers")

    missing_customers = pd.DataFrame(missing_customers["customer_id"]).assign(week=reference_week, sales_channel_id=1)

    missing_candidates = generate_candidates_single(method, missing_customers, all_transactions, transactions)
    logging.debug(f"[CANDIDATES] Generated {len(missing_candidates)} candidates for missing customers - average of {len(missing_candidates) / len(missing_customers)} per customer")
    return missing_candidates


def add_features(data, selected_features, all_transactions, transactions, customers, articles):
    """
    Add features to the data
    :param data: Transactions dataframe to add features to
    :param selected_features: List of features to add
    :param all_transactions: Full transactions dataframe
    :param transactions: Filtered transactions dataframe (everything in training interval)
    :param customers: Customers dataframe
    :param articles: Articles dataframe
    :return: Transactions dataframe with added features
    """
    if "weekly_rank" in selected_features:
        bestsellers_weekly = get_most_sold_weekly_products(transactions)
        # bestsellers_weekly['week'] += 1
        bestsellers_weekly = bestsellers_weekly.sort_values(["week", "rank"]).groupby("week").head(12)
        data = weekly_bestseller_feature(data, bestsellers_weekly)

    if "all_time_rank" in selected_features:
        bestsellers_all_time = get_most_sold_products(transactions)
        bestsellers_all_time = bestsellers_all_time.sort_values(["rank"]).head(50)
        data = all_time_bestseller_feature(data, bestsellers_all_time)

    if "price_sensitivity" in selected_features:
        data = price_sensitivity_feature(data, customers)

    if "new_arrival" in selected_features:
        data = product_age_feature(all_transactions, data)

    if "has_promotion" in selected_features:
        data = discount_feature(data)

    if "age_group" in selected_features:
        data = age_group_feature(data, customers)

    return data


def clean_up_transaction_data(data):
    """
    Clean up transaction data by dropping duplicates and ordering by week and customer_id
    :param data: Transactions dataframe
    :return: Cleaned up transactions dataframe
    """

    # Clean up incoming data
    data.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)
    data.sort_values(['week', 'customer_id'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    if "bought" not in data:
        data["bought"] = 1

    return data


def prepare_data(config, all_transactions, transactions, customers, articles, reference_week):
    """
    Generate train and test candidates, and then add features to the data
    :param config: Configuration dictionary
    :param all_transactions: Full transactions dataframe
    :param transactions: Filtered transactions dataframe
    :param customers: Customers dataframe
    :param articles: Articles dataframe
    :param reference_week: Week to generate test candidates for
    :return: Train and test data
    """
    train_candidate_customers, test_candidate_customers = get_buying_customers_candidates(transactions, reference_week)

    logging.debug(f"[PREPARATION] Found {len(train_candidate_customers)} train candidates and {len(test_candidate_customers)} test candidates")
    logging.debug(f"[PREPARATION] Amount of customers in transactions set: {transactions['customer_id'].nunique()}")

    train_candidates = generate_candidates(config["TRAIN_CANDIDATE_METHODS"], train_candidate_customers,
                                           all_transactions, transactions, customers)
    test_candidates = generate_candidates(config["TEST_CANDIDATE_METHODS"], test_candidate_customers,
                                          all_transactions, transactions, customers, reference_week)

    # For every train candidate set, list the amount of candidates and NaN values
    for i, candidate in enumerate(train_candidates):
        logging.debug(f"[PREPARATION] Train candidate {config['TRAIN_CANDIDATE_METHODS'][i]}: {candidate.shape[0]} candidates, {candidate.isna().sum().sum()} NaN values - average of {candidate.shape[0] / candidate['customer_id'].nunique()} per customer")
    for i, candidate in enumerate(test_candidates):
        logging.debug(f"[PREPARATION] Test candidate {config['TEST_CANDIDATE_METHODS'][i]}: {candidate.shape[0]} candidates, {candidate.isna().sum().sum()} NaN values - average of {candidate.shape[0] / candidate['customer_id'].nunique()} per customer")

    # Add all candidates together with the training data
    train_data = transactions.copy()
    train_data["bought"] = 1
    train_data = pd.concat([train_data] + train_candidates + test_candidates, ignore_index=True)
    train_data.fillna(0, inplace=True)

    # Drop duplicate candidates (prevent same article being bought by same customer in same week)
    train_data.drop_duplicates(["customer_id", "article_id", "week"], inplace=True)


    logging.info(f"[PREPARATION] Finished cleaning up candidate data")

    # Add features to the data (train + test candidates
    train_data = add_features(train_data, config["ADDED_FEATURES"], all_transactions, transactions, customers, articles)


    logging.info(f"[PREPARATION] Finished adding features to training data")

    # Removes the first week of the training data (since it does not have any data it can base itself on for some features)
    train_data = train_data[train_data['week'] != train_data['week'].min()]

    # Remove all features that are not in the data
    train_data = clean_up_transaction_data(train_data.copy())

    # Split up training data and added negative samples from test candidates
    train_data, test_data = split_dataframe(train_data, train_data["week"] < reference_week)

    # Join all features together
    train_data = join_all_features(train_data, articles, customers)

    test_data.drop_duplicates(["customer_id", "article_id", "sales_channel_id"], inplace=True)
    test_data = join_all_features(test_data, articles, customers)

    logging.info(f"[PREPARATION] Finished preparation of data")


    if config["VERBOSE"]:
        logging.debug(f"[PREPARATION] Train data: {train_data.shape[0]} candidates for {train_data['customer_id'].nunique()} customers, {train_data.isna().sum().sum()} NaN values")

    return train_data, test_data
