from config import VERBOSE
from util import filter_feature_data

from lightgbm.sklearn import LGBMRanker

def train_ranker(train, available_features, config):
    # Gives amount of items purchased per customer per week (used for training the ranker)
    train_bins = train.groupby(["week", "customer_id"])["article_id"].count().values
    train_X = filter_feature_data(train, available_features)
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
    test_X = filter_feature_data(test, available_features)

    test["score"] = ranker.predict(test_X)
    test.sort_values(by=["customer_id", "score"], ascending=[True, False], inplace=True)

    return test[["article_id", "customer_id", "score"]]




def filter_prediction_accuracy(predicted_candidates, min_accuracy):
    return predicted_candidates[predicted_candidates["score"] >= min_accuracy]
