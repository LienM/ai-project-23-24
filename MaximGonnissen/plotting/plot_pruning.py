import matplotlib.pyplot as plt

from pruning.prune_no_purchases import prune_no_purchases
from pruning.prune_outdated_items import prune_outdated_items
from utils.utils import load_data_from_hnm, DataFileNames


def plot_pruning() -> None:
    """
    Plots original vs. pruned customer, article, and transaction counts as a bar chart.
    """
    original_customer_df = load_data_from_hnm(DataFileNames.CUSTOMERS.replace('.csv', '.parquet'))
    original_articles_df = load_data_from_hnm(DataFileNames.ARTICLES.replace('.csv', '.parquet'),
                                              dtype={'article_id': str})
    original_transactions_df = load_data_from_hnm(DataFileNames.TRANSACTIONS_TRAIN.replace('.csv', '.parquet'),
                                                  dtype={'article_id': str})

    # Prune customers & check how many were pruned
    pruned_customers_df = original_customer_df.copy()
    pruned_customers_df = prune_no_purchases(pruned_customers_df, original_transactions_df)

    print(f"Original customer count: {len(original_customer_df)}")
    print(f"Pruned customer count: {len(pruned_customers_df)}")
    print(f'Pruned {len(original_customer_df) - len(pruned_customers_df)} customers.')
    print(
        f'Pruned {((len(original_customer_df) - len(pruned_customers_df)) / len(original_customer_df)) * 100}% of customers.')

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_customer_df), len(pruned_customers_df)])
    ax.set_ylabel('Customer count')
    ax.set_title('Original vs. pruned customer count')
    plt.show()

    # Prune articles & check how many were pruned
    pruned_articles_df = original_articles_df.copy()
    pruned_transactions_df = original_transactions_df.copy()
    pruned_articles_df, pruned_transactions_df = prune_outdated_items(pruned_articles_df, pruned_transactions_df)

    print(f"Original article count: {len(original_articles_df)}")
    print(f"Pruned article count: {len(pruned_articles_df)}")
    print(f'Pruned {len(original_articles_df) - len(pruned_articles_df)} articles.')
    print(
        f'Pruned {((len(original_articles_df) - len(pruned_articles_df)) / len(original_articles_df)) * 100}% of articles.')

    print(f"Original transaction count: {len(original_transactions_df)}")
    print(f"Pruned transaction count: {len(pruned_transactions_df)}")
    print(f'Pruned {len(original_transactions_df) - len(pruned_transactions_df)} transactions.')
    print(
        f'Pruned {((len(original_transactions_df) - len(pruned_transactions_df)) / len(original_transactions_df)) * 100}% of transactions.')

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_articles_df), len(pruned_articles_df)])
    ax.set_ylabel('Article count')
    ax.set_title('Original vs. pruned article count')
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(['Original', 'Pruned'], [len(original_transactions_df), len(pruned_transactions_df)])
    ax.set_ylabel('Transaction count')
    ax.set_title('Original vs. pruned transaction count')
    plt.show()
