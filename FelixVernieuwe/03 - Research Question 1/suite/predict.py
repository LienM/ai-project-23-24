from config import VERBOSE
from lightgbm.sklearn import LGBMRanker
import logging


def train_ranker(train, available_features, config):
    """
    Train the LGBM model on the train data
    :param train: Train data
    :param available_features: Available features to train on
    :param config: Configuration dictionary
    :return: Trained ranker model for predicting scores
    """
    # Gives amount of items purchased per customer per week (used for training the ranker)
    train_bins = train.groupby(["week", "customer_id"])["article_id"].count().values
    train_X = train[available_features]
    train_y = train["bought"]

    ranker = LGBMRanker(
        force_row_wise=True,
        objective="lambdarank",
        metric="ndcg",
        boosting_type="dart",
        n_estimators=config["N_ESTIMATORS"],
        importance_type='gain',
        verbose=10 if config["VERBOSE"] else 0,
    )

    ranker = ranker.fit(train_X, train_y, group=train_bins)

    return ranker


def score_candidates(test, available_features, ranker):
    """
    Add prediction score column for all test candidates (for customers that we have data on)
    :param ranker: Trained ranker model on train data
    :param test: Set of test candidates
    :param available_features: Available features to predict on (must be identical to the ones used in train_ranker)
    :return: Test candidates with score field added, only identifying features remain
    """
    test_X = test[available_features]

    test["score"] = ranker.predict(test_X)
    test.sort_values(by=["customer_id", "score"], ascending=False, inplace=True)

    return test[["article_id", "customer_id", "score"]]




def filter_prediction_accuracy(predicted_candidates, min_accuracy):
    return predicted_candidates[predicted_candidates["score"] >= min_accuracy]
