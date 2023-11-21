import pandas as pd


def weekly_bestseller_feature(transactions: pd.DataFrame, bestsellers_per_week_ranked: pd.DataFrame):
    transactions = pd.merge(transactions, bestsellers_per_week_ranked[['week', 'article_id', 'bestseller_rank']],
                            on=['week', 'article_id'], how='left')
    transactions['bestseller_rank'].fillna(999, inplace=True)

    return transactions


def overal_bestseller_feature(transactions: pd.DataFrame):
    # Rank the articles based on the amount of times they have been purchased
    most_purchased_articles = transactions.groupby('article_id').size().sort_values(ascending=False)
    most_purchased_articles = most_purchased_articles.reset_index()
    most_purchased_articles['alltime_rank'] = most_purchased_articles.index + 1
    most_purchased_articles = most_purchased_articles.drop(columns=0)

    transactions = pd.merge(transactions, most_purchased_articles, on='article_id', how='left')

    return transactions
