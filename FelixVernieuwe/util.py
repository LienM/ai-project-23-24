import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import hashlib

def load_image(article_id):
    return mpimg.imread('../data/images/{}/{}.jpg'.format(article_id[:3], article_id))


def display_images(images):
    fig, ax = plt.subplots(1, len(images), figsize=(20, 20))
    for i, image in enumerate(images):
        ax[i].imshow(image)
        ax[i].axis('off')
    plt.show()


def setup_seaborn():
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")


article_types = {
    'article_id': 'int64',
    'product_code': 'int32',
    'prod_name': 'string',
    'product_type_no': 'int16',
    'product_type_name': 'string',
    'product_group_name': 'string',
    'graphical_appearance_no': 'int32',
    'graphical_appearance_name': 'string',
    'colour_group_code': 'int8',
    'colour_group_name': 'string',
    'perceived_colour_value_id': 'int8',
    'perceived_colour_value_name': 'string',
    'perceived_colour_master_id': 'int8',
    'perceived_colour_master_name': 'string',
    'department_no': 'int16',
    'department_name': 'string',
    'index_code': 'string',
    'index_name': 'string',
    'index_group_no': 'int8',
    'index_group_name': 'string',
    'section_no': 'int8',
    'section_name': 'string',
    'garment_group_no': 'int16',
    'garment_group_name': 'string',
    'detail_desc': 'string'
}

customer_types = {
    'customer_id': 'string',
    'FN': 'boolean',
    'Active': 'boolean',
    'club_member_status': 'string',
    'fashion_news_frequency': 'string',
    'age': 'float32',
    'postal_code': 'string'
}

transaction_types = {
    't_dat': 'string',
    'customer_id': 'string',
    'article_id': 'int64',
    'price': 'float32',
    'sales_channel_id': 'int8'
}


def load_csv(file_name):
    if file_name == '../data/articles.csv':
        df = pd.read_csv(file_name, dtype=article_types)
    elif file_name == '../data/customers.csv':
        df = pd.read_csv(file_name, dtype=customer_types)
    elif file_name == '../data/transactions.csv':
        df = pd.read_csv(file_name, dtype=transaction_types, parse_dates=['t_dat'])
    else:
        df = pd.read_csv(file_name)
    return df


def load_parquet(file_name):
    if file_name == '../data/transactions.csv':
        df = pd.read_parquet(file_name, engine='pyarrow')
        df['t_dat'] = pd.to_datetime(df['t_dat'])
    else:
        df = pd.read_parquet(file_name, engine='pyarrow')
    return df


def convert_csv_to_parquet(file_name):
    df = load_csv(file_name)
    df.to_parquet(file_name.replace('.csv', '.parquet'), engine='pyarrow')
    os.remove(file_name)


def convert_parquet_to_csv(file_name):
    df = pd.read_parquet(file_name)
    df.to_csv(file_name.replace('.parquet', '.csv'), index=False)
    os.remove(file_name)


def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def hash_dataframe(df):
    return hashlib.sha256(df.head(100000).to_string().encode()).hexdigest()
