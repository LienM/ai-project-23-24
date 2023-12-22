import os
import kaggle


def customer_hex_id_to_int(series):
    return series.str[-16:].apply(hex_id_to_int)


def hex_id_to_int(str):
    return int(str[-16:], 16)


def split_dataframe(df, condition):
    """
    Splits a dataframe into two dataframes based on condition.
    :returns: (df[condition], df[~condition])
    """
    return df[condition], df[~condition]





def join_all_features(transactions, articles, customers):
    # Join all features together
    transactions = transactions.merge(articles, on="article_id", how="left")
    transactions = transactions.merge(customers, on="customer_id", how="left")

    return transactions


def filter_feature_data(data, available_features):
    # Train only on available features in the data, and specified features above
    available_features = [feature for feature in available_features if feature in data.columns]

    return data[available_features]


def load_image(article_id):
    return mpimg.imread('../../data/images/{}/{}.jpg'.format(article_id[:3], article_id))


def display_product(article_id):
    image = load_image(article_id)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def generate_submission_df(submission_df, predictions):
    """Generates a submission dataframe that combines a submission dataframe with predictions for each customer."""

    # Create a mapping of customer_id (int) to a list of article_ids
    prediction_dict = predictions.groupby("customer_id")["article_id"].apply(list).to_dict()

    # It is always optimal to provide at least *some* suggestion for every user
    assert len(submission_df) == len(prediction_dict), f"Submission df and predictions df must have the same length, but got {len(submission_df)} and {len(predictions)}"

    # Since hex string customer_id is non-recoverable from the predictions, we map it back to its original digest by using the submission df
    # Precalculating the mapping is not very efficient the prediction_dict as we would have to store all those strings in memory
    # 12 products * 8B * 16.463.760 rows = 1.580.520.960B (~1.6GB)
    output_df = submission_df.copy()
    output_df.prediction = [" ".join(f"0{x}" for x in prediction_dict[customer_id]) for customer_id in customer_hex_id_to_int(submission_df["customer_id"])]
    return output_df


def initialise_kaggle(kaggle_json_path):
    """Initialises the Kaggle API."""

    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_json_path
    return kaggle.api.authenticate()


def upload_to_kaggle(filepath, message):
    """Uploads a submission to Kaggle."""

    return kaggle.api.competition_submit(filepath, message, 'h-and-m-personalized-fashion-recommendations')
