from constants import all_added_features, all_base_features, feature_notation, candidate_notation
from util import generate_submission_df, initialise_kaggle, split_dataframe, hash_dataframe, filter_available_features
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
    """
    Generate metadata for this run, based on the given config
    :param config: Config object
    :param reference_week: Week to predict candidates for
    :return: Unique submission name and description for this run
    """
    submission_number = max([int(file[len(config["EXPERIMENT_NAME"]) + 1: file.find("_", len(config["EXPERIMENT_NAME"]) + 1)])
                             for file in os.listdir(config["OUTPUT_PATH"]) if file.startswith(config["EXPERIMENT_NAME"])
                             ], default=0) + 1

    removed_base_features = ",".join(["-" + feature_notation[feature] for feature in config["FILTER_BASE_FEATURES"]])
    list_added_features = ",".join([feature_notation[feature] for feature in config["ADDED_FEATURES"]])
    list_train_candidates = ",".join([candidate_notation[candidate["type"]] for candidate in config["TRAIN_CANDIDATE_METHODS"]])
    list_test_candidates = ",".join([candidate_notation[candidate["type"]] for candidate in config["TEST_CANDIDATE_METHODS"]])

    if removed_base_features != "":
        removed_base_features = f"[r_{removed_base_features}]_"
    if list_added_features != "":
        list_added_features = f"[a_{list_added_features}]_"
    if list_train_candidates != "":
        list_train_candidates = f"[tc_{list_train_candidates}]_"
    if list_test_candidates != "":
        list_test_candidates = f"[rc_{list_test_candidates}]_"

    submission_metadata = f"[t_{config['TRAIN_WEEKS']},w_{reference_week}]_{removed_base_features}{list_added_features}{list_train_candidates}{list_test_candidates}"
    submission_name = f"{config['EXPERIMENT_NAME']}_{submission_number}_submission_{submission_metadata}"
    submission_description = f"Experiment {submission_number}: {config['EXPERIMENT_DESC']} {submission_metadata}".replace("_", " ")

    return submission_name, submission_description


def single_experiment(config):
    """
    Run a single experiment with the given config
    :param config: Config object
    :return: Score of the run (online/offline depending on config)
    """
    articles = pd.read_parquet(config["DATA_PATH"] + "articles.parquet")
    all_transactions = pd.read_parquet(config["DATA_PATH"] + "transactions_train.parquet")
    customers = pd.read_parquet(config["DATA_PATH"] + "customers.parquet")

    # The week the candidates will be *generated* for
    reference_week = all_transactions["week"].max() - config["TEST_OFFSET"] - config["TEST_WEEKS"] + 1
    all_features = [feature for feature in all_base_features + config["ADDED_FEATURES"] if feature not in config["FILTER_BASE_FEATURES"]]
    submission_name, submission_description = run_metadata(config, reference_week)

    # Split dataframe in two based on reference week
    transactions, test_data = split_dataframe(all_transactions,  all_transactions["week"] < reference_week)
    # Filter test data to only include the weeks we want to evaluate against
    test_data = test_data[test_data["week"] < reference_week + config["TEST_WEEKS"]]
    # Filter train data to only include the weeks we want to train on
    train_data = transactions[transactions["week"] >= reference_week - config["TRAIN_WEEKS"]]

    # Add features and generate candidates for both the training data (and optionally: test)
    train_data, test_candidates = prepare_data(config, all_transactions, train_data, customers, articles, reference_week)

    ranker = train_ranker(train_data, filter_available_features(all_features, train_data), config)

    predicted_candidates = score_candidates(test_candidates, filter_available_features(all_features, test_candidates), ranker)
    logging.debug(f"[PREDICTION] Scored {predicted_candidates.shape[0]} candidates for {predicted_candidates['customer_id'].nunique()} customers")
    logging.info(f"[PREDICTION] Finished scoring candidates")

    if config["PREDICTION_ACCURACY"]:
        predicted_candidates = filter_prediction_accuracy(predicted_candidates, config["PREDICTION_ACCURACY"])


    missing_candidates = generate_missing_candidates(config["MISSING_CANDIDATE_METHOD"], all_transactions, transactions, customers, reference_week, predicted_candidates)

    all_candidates = pd.concat([predicted_candidates, missing_candidates], ignore_index=True)

    # For every candidate, only keep the top k articles
    all_candidates = all_candidates.groupby("customer_id").head(config["K"]).reset_index(drop=True)[["article_id", "customer_id"]]

    logging.debug(f"[CANDIDATES] Finished with {len(all_candidates)} candidates for {all_candidates['customer_id'].nunique()} customers - average of {len(all_candidates) / all_candidates['customer_id'].nunique()} per customer")

    if config["SAVE_DATAFRAME"] or not config["TEST_WEEKS"]:
        base_submission_df = pd.read_csv(config["DATA_PATH"] + "sample_submission.csv")

        submission_df = generate_submission_df(base_submission_df, all_candidates)
        submission_df.to_csv(config["OUTPUT_PATH"] + f"{submission_name}.csv.gz", index=False)

    if config["TEST_WEEKS"]:
        run_score = offline_evaluation(all_candidates, test_data)
        score = run_score
    else:
        public_score, private_score = online_evaluation(submission_name, submission_description, config)
        score = public_score

    return score





# TEST_WEEKS is always assumed to be 1
SCORES_DF_COLUMNS = [
    "TRAIN_WEEKS", "FILTER_BASE_FEATURES", "ADDED_FEATURES",
    "TRAIN_CANDIDATE_METHODS", "TEST_CANDIDATE_METHODS", "MISSING_CANDIDATE_METHOD",
    "N_ESTIMATORS", "K", "PREDICTION_ACCURACY",
    "MAP_SCORE", "PUBLIC_SCORE"
]
IDENTIFYING_COLUMNS = SCORES_DF_COLUMNS[:-2]
SCORE_COLUMNS = SCORES_DF_COLUMNS[-2:]


def log_score(config, score):
    """
    Log configuration and achieved score to scoring csv
    :param config: Config object
    :param score: Score of the run
    """


    # This entire function is terrible. I know.
    try :
        score_df = pd.read_csv(config["OUTPUT_PATH"] + "scores.csv", dtype=str)
    except:
        score_df = pd.DataFrame(columns=SCORES_DF_COLUMNS)

    experiment_identification = { column: str(config[column]) for column in IDENTIFYING_COLUMNS }

    score_column = "PUBLIC_SCORE" if config["TEST_WEEKS"] == 0 else "MAP_SCORE"
    existing_experiments = score_df[score_df[IDENTIFYING_COLUMNS].eq(pd.Series(experiment_identification)).all(axis=1)]

    if not existing_experiments.empty:
        score_df.loc[existing_experiments.index, score_column] = score
    else:
        full_experiment_identification = {**experiment_identification, **{column: 0.0 for column in SCORE_COLUMNS}}
        full_experiment_identification[score_column] = score
        score_df.loc[len(score_df)] = full_experiment_identification

    score_df.to_csv(config["OUTPUT_PATH"] + "scores.csv", index=False)



def single_experiment_save_score(config):
    score = single_experiment(config)
    if score and config["SAVE_SCORE"]:
        log_score(config, score)
    return score


def single_experiment_eval_both(config):
    new_config = config.copy()
    new_config["TEST_WEEKS"] = 1
    map_score = single_experiment_save_score(new_config)

    new_config["TEST_WEEKS"] = 0
    public_score = single_experiment_save_score(new_config)

    return map_score, public_score

if __name__ == '__main__':
    run_config = DEFAULT_CONFIG

    logging.debug("CURRENT CONFIG: " + json.dumps(run_config, indent=4))
    logging.info("KAGGLE API INITIALISED")

    # single_experiment_save_score(run_config)
    single_experiment_eval_both(run_config)