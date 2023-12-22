import logging

from candidates.bestseller import *
from candidates.new_arrivals import *
from candidates.repurchase import *
from candidates.u2u_collaborative_filtering import uucf

from data.candidate_products import get_most_sold_products, get_most_sold_weekly_products, \
    get_most_sold_weekly_age_group_products, get_first_sale_products

def generate_candidates_single(missing_candidate_method, candidate_customers, all_transactions, transactions, k=12, max_age=0):
    # GOAL: maximize the probability of the 12 recommended items for cold start users containing a relevant item
    #   Note: the highest probabilities should appear first in the list of 12 recommended items (for higher MAP)
    #               higher probability of a random user buying that item -> higher probability that it is in fact relevant
    #   Addendum: since the cold start users have no history or distinguishing purchase history features,
    #               the set of recommendations are functionally identical for each group

    # Note that we will only select a single candidate method, if we have more than 12 items, there is no way
    #     to determine which ones are better than the others (by default we just select the first 12)

    resulting_candidates = None

    if "weekly_bestsellers" == missing_candidate_method:
        bestsellers_weekly = get_most_sold_weekly_products(transactions)


        # reference_week = bestsellers_weekly['week'].max() + 1
        # bestsellers_weekly = bestsellers_weekly[bestsellers_weekly['week'] == bestsellers_weekly['week'].max()]
        # bestsellers_weekly['week'] = reference_week

        resulting_candidates = candidate_bestsellers_weekly(candidate_customers, bestsellers_weekly, k)

    elif "all_time_bestsellers" == missing_candidate_method:
        bestsellers_all_time = get_most_sold_products(all_transactions)
        resulting_candidates = candidate_bestsellers_all_time(candidate_customers, bestsellers_all_time, k)

    elif "age_group_bestsellers" == missing_candidate_method:
        bestsellers_age_group = get_most_sold_weekly_age_group_products(transactions, customers)
        resulting_candidates = candidate_bestsellers_age_group(candidate_customers, bestsellers_age_group, k)

    elif "new_arrivals" == missing_candidate_method:
        resulting_candidates = candidate_new_arrivals(candidate_customers, all_transactions, k, max_age)

    logging.info(f"[CANDIDATES] Generated {len(resulting_candidates)} candidates for {len(candidate_customers)} customers in category {missing_candidate_method} - average of {len(resulting_candidates) / len(candidate_customers)} per customer")
    return resulting_candidates


def generate_candidates(candidate_methods, candidate_customers, all_transactions, transactions, customers, k=12, max_age=0, reference_week=None):
    resulting_candidates = []

    # GOAL: for hot users (not literally), maximize the probability that the user buys item given their context

    if "all_time_bestsellers" in candidate_methods:
        bestsellers_all_time = get_most_sold_products(all_transactions)
        resulting_candidates.append(candidate_bestsellers_all_time(candidate_customers, bestsellers_all_time, k))

    if "weekly_bestsellers" in candidate_methods:
        bestsellers_weekly = get_most_sold_weekly_products(transactions)
        resulting_candidates.append(candidate_bestsellers_weekly(candidate_customers, bestsellers_weekly, k))

    if "age_group_bestsellers" in candidate_methods:
        bestsellers_age_group = get_most_sold_weekly_age_group_products(transactions, customers)
        resulting_candidates.append(candidate_bestsellers_age_group(candidate_customers, bestsellers_age_group, k))

    if "previous_purchases" in candidate_methods:
        resulting_candidates.append(candidate_previous_purchases(transactions, reference_week))

    if "new_arrivals" in candidate_methods:
        resulting_candidates.append(candidate_new_arrivals(candidate_customers, all_transactions, k, max_age))

    logging.info(f"[CANDIDATES] Generated {sum([len(candidate) for candidate in resulting_candidates])} candidates for {len(candidate_customers)} customers in categories {candidate_methods} - average of {sum([len(candidate) for candidate in resulting_candidates]) / len(candidate_customers)} per customer")
    return resulting_candidates
