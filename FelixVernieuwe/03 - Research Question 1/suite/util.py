import os
import kaggle
import logging


def customer_hex_id_to_int(series):
    """
    Converts a series of customer hex ids into integers for more memory efficient storage.
    :param series: Series of customer hex ids
    :return: Series of customer ids as integers
    """
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    """
    Convert a hex string identifier to an integer.
    :param str: Hex string identifier
    :return: Integer identifier
    """
    return int(str[-16:], 16)


def split_dataframe(df, condition):
    """
    Splits a dataframe into two dataframes based on condition.
    :returns: (df[condition], df[~condition])
    """
    return df[condition], df[~condition]


def join_all_features(transactions, articles, customers):
    """
    Joins all features together (transactions + articles + customers)
    :param transactions: Transactions dataframe
    :param articles: Articles dataframe
    :param customers: Customers dataframe
    :return: Joined dataframe
    """
    # Join all features together
    transactions = transactions.merge(articles, on="article_id", how="left")
    transactions = transactions.merge(customers, on="customer_id", how="left")

    return transactions


def filter_available_features(data, available_features):
    """
    Filters out all features that are not in the data, give a warning if this happens.
    :param data: Dataframe with features
    :param available_features: List of available features
    :return: Filtered list of features
    """

    filtered_features = [feature for feature in data if feature in available_features]
    if len(filtered_features) != len(data):
        logging.error(f"[FEATURES] Not all features are in the data: {set(data) - set(filtered_features)}")
    return filtered_features


def load_image(article_id: str):
    """
    Loads an image from the data folder for displaying in a notebook.
    :param article_id: Article id as a string
    :return: Image as a numpy array
    """

    return mpimg.imread('../../data/images/{}/{}.jpg'.format(article_id[:3], article_id))


def display_product(article_id):
    """
    Displays a product image in a notebook.
    :param article_id: Article id as a string
    :return: None
    """

    image = load_image(article_id)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def generate_submission_df(submission_df, predictions):
    """
    Generates a submission dataframe that combines a submission dataframe with predictions for each customer.
    :param submission_df: Sample submission dataframe
    :param predictions: Predictions dataframe with minified customer ids (that need to be converted back to hex strings)
    :return: Submission dataframe with predictions
    """

    # Create a mapping of customer_id (int) to a list of article_ids
    prediction_dict = predictions.groupby("customer_id")["article_id"].apply(list).to_dict()

    # It is always optimal to provide at least *some* suggestion for every user
    assert len(submission_df) == len(prediction_dict), f"Submission df and predictions df must have the same length, but got {len(submission_df)} and {len(prediction_dict)}"

    # Since hex string customer_id is non-recoverable from the predictions, we map it back to its original digest by using the submission df
    # Precalculating the mapping is not very efficient the prediction_dict as we would have to store all those strings in memory
    # 12 products * 8B * 16.463.760 rows = 1.580.520.960B (~1.6GB)
    output_df = submission_df.copy()
    output_df.prediction = [" ".join(f"0{x}" for x in prediction_dict[customer_id]) for customer_id in customer_hex_id_to_int(submission_df["customer_id"])]
    return output_df


def hash_dataframe(df, n=100000):
    """Encodes the first n rows of a dataframe as a string and returns the sha256 hash of that string."""
    return hashlib.sha256(df.head(n).to_string().encode()).hexdigest()


def initialise_kaggle(kaggle_json_path):
    """Initialises the Kaggle API."""

    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_json_path
    return kaggle.api.authenticate()


def upload_to_kaggle(filepath, message):
    """Uploads a submission to Kaggle."""

    return kaggle.api.competition_submit(filepath, message, 'h-and-m-personalized-fashion-recommendations')
