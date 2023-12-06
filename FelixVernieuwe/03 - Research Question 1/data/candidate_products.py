import pandas as pd


def get_weekly_purchase_price(transactions: pd.DataFrame):
    """Get the average price per week of a product"""
    return transactions.groupby(['week', 'article_id'])['price'].mean().reset_index()


def get_most_common_sales_channel(transactions: pd.DataFrame):
    """Get the most common sales channel of a customer per week"""
    return transactions.groupby(['week', 'customer_id'])['sales_channel_id'].apply(lambda x: x.mode().iloc[0]).reset_index()
