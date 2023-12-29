import logging

import pandas as pd
import logging


def candidate_previous_purchases(transactions: pd.DataFrame, reference_week=None):
    """
    For every customer, add the products they bought in the previous week as potential candidates
    :param transactions: Filtered transactions dataframe
    :param reference_week: Week to use as reference week (None: generate training candidates, else: generate test candidates for given reference week)
    :return: Products bought in the previous week as potential candidates for every customer that bought a product in given period
    """

    if not reference_week:
        # For every customer, lists the weeks in which they have made a purchase
        # C1234: [94, 95, 96, 101, 102] (bought an item in week 94, 95, etc.)
        customer_weekly_purchase_activity = transactions.groupby('customer_id')['week'].unique()

        # For every customer, map the purchase activity week to the next week they made a purchase, and the last week to the reference week
        #    C1234: [94, 95, 96, 101, 102]
        # -> C1234: {94: 95, 95: 96, 96: 101, 101: 102, 102: 104}
        customer_weekly_purchase_activity_shifted = {}
        for customer, weeks in customer_weekly_purchase_activity.items():
            customer_weekly_purchase_activity_shifted[customer] = {}
            for week in range(weeks.shape[0] - 1):
                customer_weekly_purchase_activity_shifted[customer][weeks[week]] = weeks[week + 1]
            customer_weekly_purchase_activity_shifted[customer][weeks[-1]] = None


        # Shift every transaction according to the shift table above
        #    idx | C1234 | A5678 | price | channel | week
        # -> idx | C1234 | A5678 | price | channel | [[week_shifted]]
        recalled_products = transactions.copy()
        recalled_products['week'] = recalled_products.apply(
            lambda row: customer_weekly_purchase_activity_shifted[row['customer_id']][row['week']], axis=1)
        recalled_products = recalled_products[recalled_products['week'].notnull()]
        recalled_products['week'] = recalled_products['week'].astype('int8')

    else:
        # For every user, get the week they last purchased an item
        last_purchase_week = transactions.groupby('customer_id')['week'].max()

        recalled_products = transactions.copy()

        # For every customer, only keep the last_purchase_week transactions
        recalled_products = recalled_products[recalled_products['week'] == recalled_products['customer_id'].map(last_purchase_week)]

        # For every customer, map the last_purchase_week to the reference week
        recalled_products['week'] = reference_week

    # End result: transactions database where items was re-purchased in a later week (when the customer bought another item)
    return recalled_products
