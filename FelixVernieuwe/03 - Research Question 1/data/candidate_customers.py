def get_buying_customers_candidates(transactions, reference_week):
    """
    Customers are included as candidate customers iff they bought any product
    :returns:
        - **candidate_customers** - for every week, all customers who bought a product in that week
        - **ref_week_candidate_customers** - all customers who ever bought a product within entire dataset
    """

    # Note: 'transaction skeleton' indicates a row with fields date, customer_id, sales_channel_id and week
    # Or in other words: a set of articles can easily be merged based on week and/or customer_id,
    #   which is equal to adding a set of candidate purchases for a given customer in a given week


    # For every customer, generate a single transaction skeleton for every week in which they bought a product
    #       customer_id | channel | week
    #       C1234       | 1       | 95
    #       C1234       | 2       | 99
    #       C5678       | 1       | 103
    candidate_customers = transactions.groupby(['week', 'customer_id']).head(1).drop(columns=['article_id', 'price']).copy()

    # For every customer, generate a single transaction skeleton in the reference week
    #       customer_id | channel | week
    #       C1234       | 1       | 104
    #       C5678       | 1       | 104
    ref_week_candidate_customers = candidate_customers.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    ref_week_candidate_customers['week'] = reference_week

    return candidate_customers, ref_week_candidate_customers


def get_all_customer_candidates(transactions, customers, reference_week):
    """
    All customers in customers dataset are included as candidate customers
    :returns:
        - **candidate_customers** - every customer appears in every week
        - **ref_week_candidate_customers** - all customers
    """

    weeks = transactions['week'].unique()

    # Gets all customer_ids from customers dataset, add every customer to every week
    candidate_customers = pd.DataFrame(list(itertools.product(customers['customer_id'].unique(), weeks)), columns=['customer_id', 'week'])
    # Randomly select channel?
    candidate_customers['channel'] = np.random.randint(1, 3, candidate_customers.shape[0])

    ref_week_candidate_customers = candidate_customers.drop_duplicates(subset=['customer_id']).reset_index(drop=True)
    ref_week_candidate_customers['week'] = reference_week
    
    return candidate_customers, ref_week_candidate_customers


def get_missing_customers_candidates(transactions, customers, reference_week):
    """
    Returns all customers that have not bought anything in the entire dataset as candidate customers
    :returns:
        - **missing_customers** - all customers that have not bought anything in the entire dataset as candidate customers
    """

    # Get all customers that have bought something in the entire dataset
    existing_customers = transactions['customer_id'].unique()

    # Get all customers that have not bought anything in the entire dataset
    missing_customers = customers[~customers["customer_id"].isin(existing_customers)][['customer_id']]

    # Polyfill a random sales channel for all missing customers
    missing_customers['channel'] = np.random.randint(1, 3, missing_customers.shape[0])

    # Set the reference week for all missing customers
    missing_customers['week'] = reference_week

    return missing_customers
