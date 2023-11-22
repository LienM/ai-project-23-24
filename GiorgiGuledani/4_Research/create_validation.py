import pandas as pd
from collections import defaultdict
from utils import customer_hex_id_to_int, path

def create_validation():
    """
    source: Radek
    Creates validation set for evaluation
    """
    articles = pd.read_parquet(f'{path}/articles.parquet')
    customers = pd.read_parquet(f'{path}/customers.parquet')
    transactions = pd.read_parquet(f'{path}/transactions_train.parquet')

    # create a validation set, which is the last week of the transactions dataset
    # source: https://github.com/radekosmulski/personalized_fashion_recs/blob/main/01_Solution_warmup.ipynb
    val_week_purchases_by_cust = defaultdict(list)

    val_week_purchases_by_cust.update(
        transactions[transactions.week == transactions.week.max()]
        .groupby('customer_id')['article_id']
        .apply(list)
        .to_dict()
    )

    sample_sub = pd.read_csv(f'{path}/sample_submission.csv')
    valid_gt = customer_hex_id_to_int(sample_sub.customer_id) \
        .map(val_week_purchases_by_cust) \
        .apply(lambda xx: ' '.join('0' + str(x) for x in xx))

    sample_sub.prediction = valid_gt
    sample_sub.to_parquet(
        f'{path}/validation_ground_truth.parquet', index=False)