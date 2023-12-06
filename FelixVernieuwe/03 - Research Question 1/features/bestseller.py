import pandas as pd


def weekly_bestseller_feature(transactions: pd.DataFrame, bestsellers_per_week_ranked: pd.DataFrame):
    transactions = pd.merge(transactions, bestsellers_per_week_ranked[['week', 'article_id', 'bestseller_rank']],
                            on=['week', 'article_id'], how='left')
    transactions['bestseller_rank'].fillna(999, inplace=True)

    return transactions


def overal_bestseller_feature(transactions: pd.DataFrame, most_purchased_articles: pd.DataFrame):

    transactions = pd.merge(transactions, most_purchased_articles, on='article_id', how='left')

    return transactions
