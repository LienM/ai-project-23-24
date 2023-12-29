import pandas as pd
from features import age_group_feature


def get_first_sale_products(transactions: pd.DataFrame):
    """Get the first sale of products for every week"""
    return transactions.groupby('article_id')['week'].min().reset_index(name='first_week')


def get_most_sold_products(transactions: pd.DataFrame):
    """Get the most sold products of all time"""
    # Return both count and rank, count by using size
    most_sold = transactions.groupby('article_id').size().reset_index(name='count')
    # Rank dataframe by count
    most_sold['rank'] = most_sold['count'].rank(method='first', ascending=False)

    # Get mean price of all time
    most_sold = pd.merge(most_sold, transactions.groupby('article_id')['price'].mean().reset_index(), on='article_id')

    return most_sold


def get_most_sold_weekly_products(transactions: pd.DataFrame):
    """Get the most sold products per week"""
    # Return both count and rank, count by using size
    most_sold = transactions.groupby(['week', 'article_id']).size().reset_index(name='count')
    # Rank dataframe by count
    most_sold['rank'] = most_sold.groupby('week')['count'].rank(method='first', ascending=False)

    # Get mean price per week
    most_sold = pd.merge(most_sold, transactions.groupby(['week', 'article_id'])['price'].mean().reset_index(), on=['week', 'article_id'])
    most_sold['week'] += 1
    # Order by week and rank
    most_sold = most_sold.sort_values(['week', 'rank'])
    most_sold.reset_index(drop=True, inplace=True)
    most_sold['rank'] = most_sold['rank'].astype('int32')

    return most_sold



def get_most_sold_weekly_age_group_products(transactions: pd.DataFrame, customers: pd.DataFrame):
    """Get the most sold products per week per age group"""

    # Add age feature to transactions
    aged_transactions = age_group_feature(transactions, customers)

    # Determine the most bought articles per age group
    top_articles_per_age_group = aged_transactions.groupby(['week', 'age', 'article_id']).size().reset_index(name='count')
    top_articles_per_age_group['rank'] = top_articles_per_age_group.groupby(['week', 'age'])['count'].rank(method='dense', ascending=False)

    return top_articles_per_age_group


def get_weekly_purchase_price(transactions: pd.DataFrame):
    """Get the average price per week of a product"""
    return transactions.groupby(['week', 'article_id'])['price'].mean().reset_index()


def get_most_common_sales_channel(transactions: pd.DataFrame):
    """Get the most common sales channel of a customer per week"""
    return transactions.groupby(['week', 'customer_id'])['sales_channel_id'].apply(lambda x: x.mode().iloc[0]).reset_index()
