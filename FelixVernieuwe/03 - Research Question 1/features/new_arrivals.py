import pandas as pd


def new_arrival_feature(transactions, data, k=5):
    first_sighting = transactions.groupby('article_id')['week'].min().reset_index()
    first_sighting = first_sighting.set_index('article_id')['week']

    # Adds the age feature to the data (age = 0 means the article was released that week)
    data['age'] = data['week'] - data['article_id'].map(first_sighting)

    # New arrivals are all product whose age is 0 in the data
    new_arrivals = data[data['age'] == 0][['week', 'article_id']].drop_duplicates().reset_index(drop=True)

    new_arrivals = new_arrivals.groupby(['week']).head(k)

    # To every customer, add the new arrivals of the week as a candidate
    data = data.merge(new_arrivals, on=['week'], how='left')

    return data