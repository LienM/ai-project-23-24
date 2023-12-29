import logging

from candidates.bestseller import *
from candidates.new_arrivals import *
from candidates.repurchase import *
from candidates.u2u_collaborative_filtering import uucf

from data.candidate_products import get_most_sold_products, get_most_sold_weekly_products, \
    get_most_sold_weekly_age_group_products, get_first_sale_products

def generate_candidates_single(method, candidate_customers, all_transactions, transactions):
    """
    For every missing candidate customer, add the output for every given method as potential candidates
    :param method: Method to use for generating candidates
    :param candidate_customers: Customers that can receive candidates for the reference week
    :param all_transactions: All transactions dataframe
    :param transactions: Filtered transactions dataframe
    :param k: Amount of candidates to generate per customer
    :return: Generated candidates for given method
    """
    # GOAL: maximize the probability of the 12 recommended items for cold start users containing a relevant item
    #   Note: the highest probabilities should appear first in the list of 12 recommended items (for higher MAP)
    #               higher probability of a random user buying that item -> higher probability that it is in fact relevant
    #   Addendum: since the cold start users have no history or distinguishing purchase history features,
    #               the set of recommendations are functionally identical for each group

    # Note that we will only select a single candidate method, if we have more than 12 items, there is no way
    #     to determine which ones are better than the others (by default we just select the first 12)

    resulting_candidates = None

    if method["type"] == "weekly_bestsellers":
        bestsellers_weekly = get_most_sold_weekly_products(transactions)
        resulting_candidates = candidate_bestsellers_weekly(candidate_customers, bestsellers_weekly, method["k"])

    elif method["type"] == "all_time_bestsellers":
        bestsellers_all_time = get_most_sold_products(all_transactions)
        resulting_candidates = candidate_bestsellers_all_time(candidate_customers, bestsellers_all_time, method["k"])

    elif method["type"] == "age_group_bestsellers":
        bestsellers_age_group = get_most_sold_weekly_age_group_products(transactions, customers)
        resulting_candidates = candidate_bestsellers_age_group(candidate_customers, customers, bestsellers_age_group, method["k"])

    elif method["type"] == "new_arrivals":
        resulting_candidates = candidate_new_arrivals(candidate_customers, all_transactions, method["max_age"], method["k"])

    logging.debug(f"[CANDIDATES] Generated {len(resulting_candidates)} candidates for {len(candidate_customers)} customers in category {method} - average of {len(resulting_candidates) / len(candidate_customers)} per customer")
    return resulting_candidates


def generate_candidates(methods, candidate_customers, all_transactions, transactions, customers, reference_week=None):
    """
    For every customer candidate, add the output for every given method as potential candidates
    :param methods: Methods to use for generating candidates
    :param candidate_customers: Customers that can receive candidates
    :param all_transactions: All transactions dataframe
    :param transactions: Filtered transactions dataframe
    :param customers: Customers dataframe
    :param reference_week: Whether to generate for train or test candidates
    :return: Generated candidates for given methods
    """
    resulting_candidates = []

    # GOAL: for hot users (not literally), maximize the probability that the user buys item given their context

    for method in methods:
        if method["type"] == "all_time_bestsellers":
            bestsellers_all_time = get_most_sold_products(all_transactions)
            resulting_candidates.append(candidate_bestsellers_all_time(candidate_customers, bestsellers_all_time, method["k"]))

        if method["type"] == "weekly_bestsellers":
            bestsellers_weekly = get_most_sold_weekly_products(transactions)
            resulting_candidates.append(candidate_bestsellers_weekly(candidate_customers, bestsellers_weekly, method["k"]))

        if method["type"] == "age_group_bestsellers":
            bestsellers_age_group = get_most_sold_weekly_age_group_products(transactions, customers)
            resulting_candidates.append(candidate_bestsellers_age_group(candidate_customers, customers, bestsellers_age_group, method["k"]))

        if method["type"] == "previous_purchases":
            resulting_candidates.append(candidate_previous_purchases(transactions, reference_week))

        if method["type"] == "new_arrivals":
            resulting_candidates.append(candidate_new_arrivals(candidate_customers, all_transactions, method["max_age"], method["k"]))

    logging.debug(f"[CANDIDATES] Generated {sum([len(candidate) for candidate in resulting_candidates])} candidates for {len(candidate_customers)} customers in categories {methods} - average of {sum([len(candidate) for candidate in resulting_candidates]) / len(candidate_customers)} per customer")
    return resulting_candidates
