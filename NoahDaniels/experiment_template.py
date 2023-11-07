import pandas as pd


def get_purchases(transactions):
    """
    Convert a dataframe containing transactions to a dataframe where each row has a customer_id and a list of purchases for that customer.

    @param transactions: a dataframe of transactions
    """
    return (
        transactions.groupby("customer_id", as_index=False)
        .article_id.apply(set)
        .rename(columns={"article_id": "purchases"})[["customer_id", "purchases"]]
    )


def get_predictions(candidates, features, ranker, k=12):
    """
    Uses a dataframe of candidates, a dataframe of features belonging to the candidates, and a trained ranker to generate k predictions for each customer represented in the candidates.
    The candidates dataframe must have the same index as the features dataframe.

    The ranker must have a predict method that takes a dataframe of features and returns a series of scores.

    @candidates: a dataframe of candidates (customer_id, article_id)
    @features: a dataframe of features belonging to the candidates
    @ranker: a trained ranker
    @k: the number of predictions to generate for each customer
    """
    scored_candidates = candidates.copy()
    scored_candidates["score"] = ranker.predict(features)

    return (
        scored_candidates.sort_values(["customer_id", "score"], ascending=False)
        .groupby("customer_id")
        .head(k)
        .groupby("customer_id", as_index=False)
        .article_id.apply(list)
        .rename(columns={"article_id": "prediction"})[["customer_id", "prediction"]]
    )


def fill_missing_predictions(predictions, customers, prediction):
    """
    Add predictions for customers that are not in the predictions dataframe.

    @param predictions: the original predictions dataframe
    @param customers: a list of customer ids for which the prediction should be added if they are missing
    @param prediction: a list of article ids that is to be used as the prediction
    """
    missing_customers = pd.Series(
        list(set(customers) - set(predictions.customer_id)),
        name="customer_id",
    )
    missing_predictions = pd.merge(
        missing_customers, pd.Series([prediction], name="prediction"), how="cross"
    )

    return pd.concat((predictions, missing_predictions))


def mean_average_precision(predictions, purchases, k=12):
    """
    Calculates the mean average precision for a set of predictions and purchases.
    Each row in the predictions and purchases has a customer_id and a list of purchases or predictions.

    @param predictions: a dataframe of predictions
    @param purchases: a dataframe of ground truth purchases
    """

    def average_precision(row):
        score = 0
        num_hits = 0

        for i, p in enumerate(row.prediction[:k]):
            if p in row.purchases and p not in row.prediction[:i]:
                num_hits += 1
                score += num_hits / (i + 1)

        return score / min(len(row.purchases), k)

    result = pd.merge(purchases, predictions, on="customer_id", how="inner")
    result["average_precision"] = result.apply(average_precision, axis=1)

    return result.average_precision.sum() / len(purchases)


def create_submission(predictions, sample_submission):
    predictions = predictions.set_index("customer_id").prediction.to_dict()
    preds = []
    result = sample_submission.copy()
    for customer_id in customer_hex_id_to_int(result.customer_id):
        preds.append(" ".join(f"0{x}" for x in predictions[customer_id]))
    result.prediction = preds
    return result


# https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/discussion/308635
def customer_hex_id_to_int(series):
    def hex_id_to_int(str):
        return int(str[-16:], 16)

    return series.str[-16:].apply(hex_id_to_int)


def print_importance(ranker, features):
    for i in ranker.feature_importances_.argsort()[::-1]:
        imp = ranker.feature_importances_[i] / ranker.feature_importances_.sum()
        print(f"{features[i]:>30} {imp:.5f}")
