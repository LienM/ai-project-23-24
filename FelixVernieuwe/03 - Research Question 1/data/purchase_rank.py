import pandas as pd


def most_sold_per_week(transactions: pd.DataFrame):
    """For every week, get the top-12 most sold products"""

    # For every article, get the mean price it was sold for per week
    # week | (article_id, price)
    #   95 |
    #           A1234 | 0.15
    #           A5678 | 0.20
    mean_product_price_per_week = transactions.groupby(['week', 'article_id'])['price'].mean()

    # For every article, get the amount of times it was sold per week
    # week | (article_id, count)
    #   95 |
    #           A1234 | 100
    #           A5678 | 500
    most_sold_products_per_week = transactions.groupby('week')['article_id'].value_counts()

    # For every week, get the top 12 most sold products and rank them under the column 'bestseller_rank'
    # week | (article_id, bestseller_rank)
    #   95 |
    #           A5678 | 1
    #           A1234 | 2
    most_sold_products_per_week_ranked = most_sold_products_per_week \
        .groupby('week').rank(ascending=False, method='dense') \
        .groupby('week').head(12).rename('bestseller_rank').astype('int8')

    # Combine the mean price and the bestseller rank
    # week | article_id | bestseller_rank | price
    #   95 | A5678      | 1               | 0.20
    #   95 | A1234      | 2               | 0.15
    most_sold_products_per_week_ranked = pd.merge(most_sold_products_per_week_ranked, mean_product_price_per_week,
                                                  on=['week', 'article_id']).reset_index()
    # Shift the week by 1 (prepare the transaction as a possible candidate)
    most_sold_products_per_week_ranked['week'] += 1

    # End result: top-12 best ranked articles per week in transaction-format (to be merged with customer)
    return most_sold_products_per_week_ranked


def most_sold_all_time(transactions: pd.DataFrame):
    """Get the most sold products all time"""

    # Rank the articles based on the amount of times they have been purchased
    most_purchased_articles = transactions.groupby('article_id').size().sort_values(ascending=False)
    most_purchased_articles = most_purchased_articles.reset_index()
    most_purchased_articles['all_time_rank'] = most_purchased_articles.index + 1
    most_purchased_articles = most_purchased_articles.drop(columns=0)

    return most_purchased_articles
