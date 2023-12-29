import json
import time
import logging
import kaggle
import pandas as pd
from scorers import mean_average_precision


def offline_evaluation(predicted_candidates, ground_truth_purchases):
    """
    Evaluate the predicted candidates against the ground truth purchases using MAP@k
    :param predicted_candidates: Dataframe of predicted candidates
    :param ground_truth_purchases: Dataframe of ground truth purchases
    :return: MAP@k score
    """

    assert len(predicted_candidates) > 0, "No candidates were predicted, cannot evaluate"
    assert len(ground_truth_purchases) > 0, "No ground truth purchases were given, cannot evaluate"


    # Get dataframe of customers and their bought articles, grouped by customer and as set, for convenient comparison operations
    ground_truth_purchases = ground_truth_purchases.groupby("customer_id")["article_id"].apply(set).reset_index()
    predicted_purchases = predicted_candidates.groupby("customer_id")["article_id"].apply(list).reset_index()

    # Rename column to match the predicted candidates
    ground_truth_purchases.rename(columns={"article_id": "purchases"}, inplace=True)
    predicted_purchases.rename(columns={"article_id": "predictions"}, inplace=True)

    truth_prediction = mean_average_precision(predicted_purchases, ground_truth_purchases)
    logging.info(f"[PREDICTION] MAP@k: {truth_prediction}")

    return truth_prediction


def online_evaluation(name, description, config):
    """
    Upload the predictions to kaggle and retrieve the score (requires Kaggle API to be installed, configured and authenticated)
    :param name: Name of the file to upload
    :param description: Description of the file to upload
    :param config: Configuration dictionary
    :return: Public and private score
    """
    result = kaggle.api.competition_submit(config["OUTPUT_PATH"] + f"{name}.csv.gz", description, 'h-and-m-personalized-fashion-recommendations')
    logging.debug(f"[PREDICTION] KAGGLE UPLOAD RESULT: {json.dumps(result.__dict__, indent=4)}")

    # Sleep at minimum 15 seconds to allow Kaggle to process the submission
    time.sleep(15)

    def attempt_score_retrieval():
        """
        Retrieve the public and private score from Kaggle
        """
        submissions = kaggle.api.competitions_submissions_list("h-and-m-personalized-fashion-recommendations")
        return submissions[0]["publicScore"], submissions[0]["privateScore"]

    # Maximal 5 attempts to retrieve the score (with 20 seconds in between to not spam the API)
    public_score, private_score = 0.0, 0.0
    for i in range(4):
        public_score, private_score = attempt_score_retrieval()
        if public_score != 0.0 and private_score != 0.0:
            break
        time.sleep(20)

    logging.info(f"[PREDICTION] KAGGLE SCORE: {public_score} (public) - {private_score} (private)")

    return public_score, private_score