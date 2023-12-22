import json
import time
import logging
import kaggle
import pandas as pd

def offline_evaluation(predicted_candidates, test_data, reference_week):
    # Filter test_data to only the reference week
    ground_truth_purchases = test_data[test_data["week"] > reference_week]

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
    result = kaggle.api.competition_submit(config["OUTPUT_PATH"] + f"{name}.csv.gz", description, 'h-and-m-personalized-fashion-recommendations')
    logging.debug(f"[PREDICTION] KAGGLE UPLOAD RESULT: {json.dumps(result.__dict__, indent=4)}")

    # Get kaggle score for this submission via API
    time.sleep(15)

    def attempt_score_retrieval():
        submissions = kaggle.api.competitions_submissions_list("h-and-m-personalized-fashion-recommendations")
        return submissions[0]["publicScore"], submissions[0]["privateScore"]

    # Maximal 3 attempts to retrieve the score
    public_score, private_score = 0.0, 0.0
    for i in range(3):
        public_score, private_score = attempt_score_retrieval()
        if public_score != 0.0 and private_score != 0.0:
            break
        time.sleep(15)

    logging.info(f"[PREDICTION] KAGGLE SCORE: {public_score} (public) - {private_score} (private)")

    return public_score, private_score