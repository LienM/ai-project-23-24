import pandas as pd
import seaborn as sns
from matplotlib import ticker

# Percentage of sales within a week that should be discounted to be considered a promotion
PROMOTION_THRESHOLD = 0.6


def discount_feature(transactions: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame):
    maximum_product_price = transactions.groupby(['week', 'article_id'])['price'].max().reset_index()

    transactions = transactions.merge(maximum_product_price, on=['week', 'article_id'], suffixes=('', '_max'))
    transactions['discount'] = (transactions['price_max'] - transactions['price']) / transactions['price_max']

    transactions['has_promotion'] = transactions['discount'] != 0
    promoted_articles = transactions.groupby(['week', 'article_id'])['has_promotion'].mean().reset_index()
    promoted_articles['has_promotion'] = promoted_articles['has_promotion'] >= PROMOTION_THRESHOLD

    return transactions, customers, articles, promoted_articles
    
    
def plot_weekly_promo_percentage(promoted_articles: pd.DataFrame):
    discounted_transactions = promoted_articles.groupby('week')['has_promotion'].mean().reset_index()

    # Plot as a stacked bar chart
    sns.barplot(x='week', y='has_promotion', data=discounted_transactions)


def plot_product_sales(product, transactions: pd.DataFrame, promoted_articles: pd.DataFrame):
    # Get the amount of purchases over time of a given article
    product_purchases = transactions[transactions['article_id'] == product]['week'].value_counts().sort_index()

    # Gets the discount information per week for the product
    product_discount = promoted_articles[promoted_articles['article_id'] == product].set_index('week')['has_promotion']
    product_purchases = product_purchases.to_frame().join(product_discount).fillna(False)

    # Plots the data, a bar is red if there is no discount, green if there is a discount
    plot = sns.barplot(x=product_purchases.index, y='week', data=product_purchases, hue='has_promotion', palette=['red', 'green'])

    plot.set_xlabel('Week')
    plot.set_ylabel('Sales')

    handles, labels = plot.get_legend_handles_labels()
    plot.legend(handles, ['No promo', 'Promotion'])

    # Set minimum bar width
    for patch in plot.patches:
        patch.set_width(0.4)

    plot.xaxis.set_major_locator(ticker.MultipleLocator(4))
    plot = plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    plot.legend(title='Discount', loc='upper left', labels=['No discount', 'Discount'])


def plot_random_product_sales(transactions: pd.DataFrame, promoted_articles: pd.DataFrame):
    # Sample a random article_id that has more than 1000 transactions
    product = transactions['article_id'].value_counts()[transactions['article_id'].value_counts() > 5000].sample(1).index[0]

    plot_product_sales(product, transactions, promoted_articles)