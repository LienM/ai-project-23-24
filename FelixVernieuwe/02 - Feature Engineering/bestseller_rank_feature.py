import pandas as pd


def bestseller_rank_feature(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, reference_week):
    transactions = transactions[transactions['week'] > reference_week - 11]

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
    # ISSUE: This is never assigned in the original code
    transactions.drop_duplicates(['week', 'customer_id'])

    candidate_best_sellers = pd.merge(unique_transactions, most_sold_products_per_week_ranked, on='week')

    reference_week_transactions = unique_transactions.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    reference_week_transactions['week'] = reference_week

    candidate_best_sellers_reference_week = pd.merge(reference_week_transactions, most_sold_products_per_week_ranked,
                                                     on='week')

    all_candidate_best_sellers = pd.concat([candidate_best_sellers, candidate_best_sellers_reference_week])
    all_candidate_best_sellers.drop(columns=['bestseller_rank'], inplace=True)

    transactions['bought'] = 1

    transactions = pd.concat([transactions, data_shifted, all_candidate_best_sellers])
    transactions.fillna(0, inplace=True)

    transactions.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)
    transactions = pd.merge(transactions, most_sold_products_per_week_ranked[['week', 'article_id', 'bestseller_rank']],
                            on=['week', 'article_id'], how='left')

    # Remove the oldest data
    first_week = transactions['week'].min()
    transactions = transactions[transactions['week'] != first_week]

    transactions['bestseller_rank'].fillna(999, inplace=True)

    return transactions, customers, articles
