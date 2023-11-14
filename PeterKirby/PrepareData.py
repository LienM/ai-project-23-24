import numpy as np
import pandas as pd

#prepare_data does all the data preparation stage (from Radek's notebook)
def prepare_data(kaggle_submission=True):

    #Loading data

    all_transactions = pd.read_parquet('../../../Data/parquets/transactions_train.parquet')
    customers = pd.read_parquet('../../../Data/parquets/customers.parquet')
    articles = pd.read_parquet('../../../Data/parquets/articles.parquet')


    if kaggle_submission:
        test_week = all_transactions.week.max() + 1
        transactions = all_transactions[all_transactions.week > all_transactions.week.max() - 10]
    else:
        test_week = all_transactions.week.max()
        transactions = all_transactions[(all_transactions.week > all_transactions.week.max() - 11) & (all_transactions.week != test_week)]



    #Generating candidates (last purchase)

    c2weeks = transactions.groupby('customer_id')['week'].unique()


    c2weeks2shifted_weeks = {}
    for c_id, weeks in c2weeks.items():
        c2weeks2shifted_weeks[c_id] = {}
        for i in range(weeks.shape[0]-1):
            c2weeks2shifted_weeks[c_id][weeks[i]] = weeks[i+1]
        c2weeks2shifted_weeks[c_id][weeks[-1]] = test_week


    candidates_last_purchase = transactions.copy()


    weeks = []
    for i, (c_id, week) in enumerate(zip(transactions['customer_id'], transactions['week'])):
        weeks.append(c2weeks2shifted_weeks[c_id][week])
    candidates_last_purchase.week=weeks



    #Generating candidates (bestsellers)

    mean_price = transactions.groupby(['week', 'article_id'])['price'].mean()


    sales = transactions \
        .groupby('week')['article_id'].value_counts() \
        .groupby('week').rank(method='dense', ascending=False) \
        .groupby('week').head(12).rename('bestseller_rank').astype('int8')


    bestsellers_previous_week = pd.merge(sales, mean_price, on=['week', 'article_id']).reset_index()
    bestsellers_previous_week.week += 1


    bestsellers_previous_week.pipe(lambda df: df[df['week']==96])


    unique_transactions = transactions \
        .groupby(['week', 'customer_id']) \
        .head(1) \
        .drop(columns=['article_id', 'price']) \
        .copy()


    candidates_bestsellers = pd.merge(
        unique_transactions,
        bestsellers_previous_week,
        on='week',
    )


    test_set_transactions = unique_transactions.drop_duplicates('customer_id').reset_index(drop=True)
    test_set_transactions.week = test_week


    candidates_bestsellers_test_week = pd.merge(
        test_set_transactions,
        bestsellers_previous_week,
        on='week'
    )


    candidates_bestsellers = pd.concat([candidates_bestsellers, candidates_bestsellers_test_week])
    candidates_bestsellers.drop(columns='bestseller_rank', inplace=True)



    #Combining transactions and candidates/negative examples

    transactions['purchased'] = 1


    data = pd.concat([transactions, candidates_last_purchase, candidates_bestsellers])
    data.purchased.fillna(0, inplace=True)


    data.drop_duplicates(['customer_id', 'article_id', 'week'], inplace=True)



    #Adding bestseller information
    data = pd.merge(
        data,
        bestsellers_previous_week[['week', 'article_id', 'bestseller_rank']],
        on=['week', 'article_id'],
        how='left'
    )


    data = data[data.week != data.week.min()]
    data.bestseller_rank.fillna(999, inplace=True)


    data = pd.merge(data, articles, on='article_id', how='left')
    data = pd.merge(data, customers, on='customer_id', how='left')


    data.sort_values(['week', 'customer_id'], inplace=True)
    data.reset_index(drop=True, inplace=True)


    train = data[data.week != test_week]
    test = data[data.week==test_week].drop_duplicates(['customer_id', 'article_id', 'sales_channel_id']).copy()


    train_baskets = train.groupby(['week', 'customer_id'])['article_id'].count().values


    if kaggle_submission:
        return train, test, train_baskets, bestsellers_previous_week
    
    else:
        return train, test, train_baskets, bestsellers_previous_week, all_transactions[all_transactions.week == test_week]

