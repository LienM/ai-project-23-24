import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import hashlib
import kaggle


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




def hash_dataframe(df, n=100000):
    """Encodes the first n rows of a dataframe as a string and returns the sha256 hash of that string."""
    return hashlib.sha256(df.head(n).to_string().encode()).hexdigest()


def compare_dataframes(df1, df2, n=100000):
    """Compares the first n rows of two dataframes and returns True if they are equal.

    NOTE: useful for quickly checking if two dataframes are *probably* the same
    """
    return hash_dataframe(df1, n) == hash_dataframe(df2, n)


def split_dataframe(df, condition):
    """
    Splits a dataframe into two dataframes based on condition.
    :returns: (df[condition], df[~condition])
    """
    return df[condition], df[~condition]


def initialise_kaggle(kaggle_json_path):
    """Initialises the Kaggle API."""

    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_json_path
    kaggle.api.authenticate()


def upload_to_kaggle(filepath, message):
    """Uploads a submission to Kaggle."""

    return kaggle.api.competition_submit(filepath, message, 'h-and-m-personalized-fashion-recommendations')


def generate_submission_df(submission_df, predictions):
    """Generates a submission dataframe that combines a submission dataframe with predictions for each customer."""

    prediction_dict = predictions.groupby("customer_id")["article_id"].apply(list).to_dict()
    output_df = submission_df.copy()
    output_df.prediction = [" ".join(f"0{x}" for x in prediction_dict[customer_id]) for customer_id in customer_hex_id_to_int(submission_df["customer_id"]) if customer_id in prediction_dict]
    return output_df


def feature_importance_df(ranker, available_features):
    """Converts the LGBM's ranking of feature importance to percentual notation"""
    # Percentual importance of each feature
    feature_importance = pd.DataFrame(
        {
            "feature": available_features,
            "importance": ranker.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    feature_importance["importance"] = feature_importance["importance"] / feature_importance["importance"].sum() * 100
    feature_importance["feature"] = feature_importance["feature"].map(lambda x: x.replace("_", " ").title())

    return feature_importance


def graph_feature_importance(feature_importance):
    """Creates a graph displaying the LGBM's ranking of feature importance."""

    feature_importance = feature_importance[feature_importance["importance"] > 0]

    plot = sns.barplot(x="importance", y="feature", data=feature_importance)
    plot.set_xlabel("Importance")
    plot.set_ylabel("Feature")

    plot.set_xscale("log")
    plot.set_xticklabels(["{:.1f}%".format(x) for x in plot.get_xticks()])
    plot.set_title("Feature Importance")

    return plot