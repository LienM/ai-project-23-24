import pandas as pd


def all_time_rank_feature(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame):
    # Rank the articles based on the amount of times they have been purchased
    most_purchased_articles = transactions.groupby('article_id').size().sort_values(ascending=False)
    most_purchased_articles = most_purchased_articles.reset_index()
    most_purchased_articles['rank'] = most_purchased_articles.index + 1
    most_purchased_articles = most_purchased_articles.drop(columns=0)

    transactions['all_time_rank'] = transactions['article_id'].map(most_purchased_articles.set_index('article_id')['rank'])

    return transactions, customers, articles
