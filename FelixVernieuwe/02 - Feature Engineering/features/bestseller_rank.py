import pandas as pd


def bestseller_rank_feature(transactions: pd.DataFrame, reference_week):
    """
    Determine the weekly bestseller rank for every product and add it to the transactions DataFrame.

    :param transactions: (filtered) transactions dataframe
    :return: transactions with bestseller rank
    """
    # Gets the weeks when the customers have bought a product
    customer_weekly_purchase_activity = transactions.groupby('customer_id')['week'].unique()

    # Get a shift table for the weeks
    customer_weekly_purchase_activity_shifted = {}
    for customer, weeks in customer_weekly_purchase_activity.items():
        customer_weekly_purchase_activity_shifted[customer] = {}
        for week in range(weeks.shape[0] - 1):
            customer_weekly_purchase_activity_shifted[customer][weeks[week]] = weeks[week + 1]
        customer_weekly_purchase_activity_shifted[customer][weeks[-1]] = reference_week

    # Shift the transactions data
    data_shifted = transactions.copy()
    data_shifted['week'] = data_shifted.apply(
        lambda row: customer_weekly_purchase_activity_shifted[row['customer_id']][row['week']], axis=1)

    # Get the mean price per week per product
    mean_product_price_per_week = transactions.groupby(['week', 'article_id'])['price'].mean()

    # Get the most frequently sold products per week and rank them
    most_sold_products_per_week = transactions.groupby('week')['article_id'].value_counts()
    most_sold_products_per_week_ranked = most_sold_products_per_week \
        .groupby('week').rank(ascending=False, method='dense') \
        .groupby('week').head(12).rename('bestseller_rank').astype('int8')

    # Merge most sold products with mean price of the next week
    most_sold_products_per_week_ranked = pd.merge(most_sold_products_per_week_ranked, mean_product_price_per_week,
                                                  on=['week', 'article_id']).reset_index()
    most_sold_products_per_week_ranked['week'] += 1

    # Get all the transactions occurring in week 95 (first week of the dataset)
    unique_transactions = transactions.groupby(['week', 'customer_id']).head(1).drop(
        columns=['article_id', 'price']).copy()

    # Drop all transactions where the customer has bought multiple products in the same week
    # ISSUE: This is never assigned in the original code (now commented)
    # transactions.drop_duplicates(['week', 'customer_id'])

    # Gets the candidate bestsellers for the reference week
    candidate_best_sellers = pd.merge(unique_transactions, most_sold_products_per_week_ranked, on='week')

    # Gets the transactions for the reference week
    reference_week_transactions = unique_transactions.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    reference_week_transactions['week'] = reference_week

    # Gets the candidate bestsellers for the reference week
    candidate_best_sellers_reference_week = pd.merge(reference_week_transactions, most_sold_products_per_week_ranked, on='week')

    # Gets all the candidate bestsellers
    all_candidate_best_sellers = pd.concat([candidate_best_sellers, candidate_best_sellers_reference_week])
    all_candidate_best_sellers.drop(columns=['bestseller_rank'], inplace=True)

    # Mark all current transactions as bought
    output = transactions.copy()
    output['bought'] = 1

    # Add all candidates as negative examples of data
    output = pd.concat([output, data_shifted, all_candidate_best_sellers])
    output.fillna(0, inplace=True)

    # Remove accidental duplicates and merge with most sold products per week (to get the bestseller rank)
    output.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)
    output = pd.merge(output, most_sold_products_per_week_ranked[['week', 'article_id', 'bestseller_rank']],
                            on=['week', 'article_id'], how='left')

    # Remove the oldest data
    first_week = output['week'].min()
    output = output[output['week'] != first_week]

    # Fill in all missing bestseller ranks with 999
    output['bestseller_rank'].fillna(999, inplace=True)

    return output
