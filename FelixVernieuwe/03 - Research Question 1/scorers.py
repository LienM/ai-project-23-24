def average_precision_score(row, k=12):
    """
    Calculates the average precision for a set of predictions.

    :param row: a row of a dataframe containing a list of predictions and a list of ground truth purchases
    :param k: the number of predictions to consider
    """
    score = 0
    n = 0

    for i, p in enumerate(row["predictions"][:k]):
        if p in row["purchases"] and p not in row["predictions"][:i]:
            n += 1
            score += n / (i + 1)

    return score / min(len(row["purchases"]), k)


def mean_average_precision(predictions, truth, k=12):
    """
    Calculates the mean average precision for a set of predictions.

    :param predictions: a dataframe containing of (customer_id, predictions) pairs
    :param truth: a dataframe of (customer_id, purchases) pairs, ground truth to compare against
    :param k: the number of predictions to consider
    """

    # Join the predictions and ground truth on customer_id
    truth_prediction = truth.merge(predictions, on="customer_id", how="inner")
    truth_prediction["average_precision@k"] = truth_prediction.apply(average_precision_score, axis=1, k=k)

    # Return the mean average precision over all customers
    return truth_prediction["average_precision@k"].sum() / len(truth)