from constants import all_added_features, all_base_features, feature_notation, candidate_notation
from util import generate_submission_df, initialise_kaggle, split_dataframe
from config import DEFAULT_CONFIG

from preparation import prepare_data, generate_missing_candidates
from predict import train_ranker, score_candidates, filter_prediction_accuracy
from evaluate import offline_evaluation, online_evaluation

import pandas as pd
import os

import logging
import sys
import json

logging.info("Starting experiment")



def run_metadata(config, reference_week):
    submission_number = max([int(file[len(config["EXPERIMENT_NAME"]) + 1: file.find("_", len(config["EXPERIMENT_NAME"]) + 1)])
                             for file in os.listdir(config["OUTPUT_PATH"]) if file.startswith(config["EXPERIMENT_NAME"])
                             ], default=0) + 1

    removed_base_features = ",".join(["-" + feature_notation[feature] for feature in config["FILTER_BASE_FEATURES"]])
    list_added_features = ",".join([feature_notation[feature] for feature in config["ADDED_FEATURES"]])
    list_training_candidates = ",".join([candidate_notation[candidate] for candidate in config["TRAIN_INTERVAL_CANDIDATES"]])
    list_reference_candidates = ",".join([candidate_notation[candidate] for candidate in config["REFERENCE_WEEK_CANDIDATES"]])

    if removed_base_features != "":
        removed_base_features = f"[r_{removed_base_features}]_"
    if list_added_features != "":
        list_added_features = f"[a_{list_added_features}]_"
    if list_training_candidates != "":
        list_training_candidates = f"[tc_{list_training_candidates}]_"
    if list_reference_candidates != "":
        list_reference_candidates = f"[rc_{list_reference_candidates}]_"

    submission_metadata = f"[t_{config['TRAINING_INTERVAL']},w_{reference_week}]_{removed_base_features}{list_added_features}{list_training_candidates}{list_reference_candidates}"
    submission_name = f"{config['EXPERIMENT_NAME']}_{submission_number}_submission_{submission_metadata}"
    submission_description = f"Experiment {submission_number}: {config['EXPERIMENT_DESC']} {submission_metadata}".replace("_", " ")

    return submission_name, submission_description


def single_run(config):
    articles = pd.read_parquet(config["DATA_PATH"] + "articles.parquet")
    all_transactions = pd.read_parquet(config["DATA_PATH"] + "transactions_train.parquet")
    customers = pd.read_parquet(config["DATA_PATH"] + "customers.parquet")

    reference_week = all_transactions["week"].max() - config["AMOUNT_OF_TEST_WEEKS"] + 1
    all_features = [feature for feature in all_base_features + config["ADDED_FEATURES"] if feature not in config["FILTER_BASE_FEATURES"]]
    submission_name, submission_description = run_metadata(config, reference_week)


    transactions = all_transactions[all_transactions["week"] >= reference_week - config["TRAINING_INTERVAL"]]
    train_data, test_data = split_dataframe(transactions,  transactions["week"] <= reference_week)

    # Add features and generate candidates for both the training data (and optionally: test)
    train_data, test_candidates = prepare_data(config, all_transactions, train_data, customers, articles, reference_week)

    ranker = train_ranker(train_data, all_features, config)
    predicted_candidates = score_candidates(test_candidates, all_features, ranker)
    logging.info(f"[PREDICTION] Scored {predicted_candidates.shape[0]} candidates for {predicted_candidates['customer_id'].nunique()} customers")


    if config["PREDICTION_ACCURACY"]:
        predicted_candidates = filter_prediction_accuracy(predicted_candidates, config["PREDICTION_ACCURACY"])


    missing_candidates = generate_missing_candidates(config["MISSING_CANDIDATE_METHOD"], all_transactions, transactions, customers,
                                                     reference_week, k=config["K_MISSING"], max_age=config["MAX_AGE"])

    all_candidates = pd.concat([predicted_candidates, missing_candidates], ignore_index=True)

    # For every candidate, only keep the top k articles
    all_candidates = all_candidates.groupby("customer_id").head(config["K"])

    logging.debug(f"[CANDIDATES] Finished with {len(all_candidates)} candidates for {all_candidates['customer_id'].nunique()} customers - average of {len(all_candidates) / all_candidates['customer_id'].nunique()} per customer")

    if config["SAVE_DATAFRAME"] or not config["AMOUNT_OF_TEST_WEEKS"]:
        base_submission_df = pd.read_csv(config["DATA_PATH"] + "sample_submission.csv")

        submission_df = generate_submission_df(base_submission_df, all_candidates)
        submission_df.to_csv(config["OUTPUT_PATH"] + f"{submission_name}.csv.gz", index=False)

    score = 0
    if config["AMOUNT_OF_TEST_WEEKS"]:
        run_score = offline_evaluation(all_candidates, test_data, reference_week)
        score = run_score
    else:
        public_score, private_score = online_evaluation(submission_name, submission_description, config)
        score = public_score

    return score


if __name__ == '__main__':

    run_config = DEFAULT_CONFIG

    logging.info("CURRENT CONFIG: " + json.dumps(run_config, indent=4))
    logging.info("KAGGLE API INITIALISED")

    # Run the experiment
    run_score = single_run(run_config)