import pandas as pd
import seaborn as sns
from matplotlib import ticker

# Percentage of sales within a week that should be discounted to be considered a promotion
PROMOTION_THRESHOLD = 0.6
RED_SHADES = ["#FFCDD2", "#EF9A9A", "#E57373", "#EF5350", "#F44336", "#E53935", "#D32F2F", "#C62828", "#B71C1C"]
GREEN_SHADES = ['#5ae064', '#4cd855', '#3fc346', '#32b737', '#259c28', '#1a8219', '#0d670a', '#005c00', '#005000']

sns.set(rc={'figure.figsize': (15, 10)})
sns.set_style("whitegrid")


def discount_data(transactions: pd.DataFrame, promotion_threshold=PROMOTION_THRESHOLD):
    """
    For every week in the transactions history, determine for every article whether it was discounted
    :param transactions: Transactions in the training period
    :param promotion_threshold: Percentage of sales within a week that should be discounted to be considered a promotion
    :return:
    """
    maximum_product_price = transactions.groupby(['week', 'article_id'])['price'].max().reset_index()

    transactions = transactions.merge(maximum_product_price, on=['week', 'article_id'], suffixes=('', '_max'))
    transactions['discount'] = (transactions['price_max'] - transactions['price']) / transactions['price_max']

    transactions['has_promotion'] = transactions['discount'] != 0
    promoted_articles = transactions.groupby(['week', 'article_id'])['has_promotion'].mean().reset_index()
    promoted_articles['has_promotion'] = promoted_articles['has_promotion'] >= promotion_threshold

    # Get most common discount per product per week
    # most_common_discount = transactions[transactions['has_promotion']] \
    #     .groupby(['week', 'article_id'])['discount'] \
    #     .agg(lambda x: x.mode().mean())
    # promoted_articles = promoted_articles.merge(most_common_discount, on=['week', 'article_id'], how='left')
    # promoted_articles['discount'] = promoted_articles['discount'].fillna(0)

    # Get average discount per product per week
    average_discount = transactions[transactions['has_promotion']] \
        .groupby(['week', 'article_id'])['discount'] \
        .mean()
    promoted_articles = promoted_articles.merge(average_discount, on=['week', 'article_id'], how='left')
    promoted_articles['discount'] = promoted_articles['discount'].fillna(0)

    return promoted_articles


def discount_feature(transactions: pd.DataFrame, promotion_threshold=PROMOTION_THRESHOLD):
    """
    For every transaction, add a feature indicating whether the product was discounted when bought
    :param transactions: Transactions in the training period
    :param promotion_threshold: Percentage of sales within a week that should be discounted to be considered a promotion
    :return: Transactions with 'has_promotion' feature added
    """
    promoted_articles = discount_data(transactions, promotion_threshold)

    # Add the discount feature to the transactions
    transactions = transactions.merge(promoted_articles, on=['week', 'article_id'], how='left')

    return transactions


def plot_weekly_promo_percentage(promoted_articles: pd.DataFrame):
    discounted_transactions = promoted_articles.groupby('week')['has_promotion'].mean().reset_index()

    # Plot as a stacked bar chart
    sns.barplot(x='week', y='has_promotion', data=discounted_transactions)


def plot_product_sales(product, transactions: pd.DataFrame, promoted_articles: pd.DataFrame):
    # Get the amount of purchases over time of a given article
    product_purchases = transactions[transactions['article_id'] == product]['week'].value_counts().sort_index()

    # Gets the discount information per week for the product
    product_discount = promoted_articles[promoted_articles['article_id'] == product].set_index('week')[
        ['has_promotion', 'discount']]
    product_purchases = product_purchases.to_frame().join(product_discount).fillna(False)

    product_purchases = product_purchases.reset_index()
    product_purchases["has_promotion"] = product_purchases["has_promotion"].astype(int).astype(str)

    # Plots the data, a bar is red if there is no discount, green if there is a discount
    plot = sns.barplot(x='week', y='count', data=product_purchases, hue='has_promotion',
                       palette=['red', 'green'])

    # Fugly way to apply shading on hue (is there a better way?)
    for bar, (promotion, discount) in zip(plot.patches, product_purchases[['has_promotion', 'discount']].values):
        if promotion == '1':
            bar.set_color(GREEN_SHADES[round(discount * 10)])
        else:
            bar.set_color(RED_SHADES[round(discount * 10)])

    plot.set_xlabel('Week')
    plot.set_ylabel('Sales')

    plot.set_title('Sales of product {} over time'.format(product))

    handles, labels = plot.get_legend_handles_labels()
    plot.legend(handles, ['No promo', 'Promotion'])

    # Set minimum bar width
    for patch in plot.patches:
        patch.set_width(0.4)

    plot.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plot = plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')


def plot_random_product_sales(transactions: pd.DataFrame, promoted_articles: pd.DataFrame):
    # Sample a random article_id that has more than 1000 transactions
    product = \
        transactions['article_id'].value_counts()[transactions['article_id'].value_counts() > 5000].sample(1).index[0]

    plot_product_sales(product, transactions, promoted_articles)
