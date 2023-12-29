import pandas as pd
import pytest

from experiment_template import *


def test_get_purchases():
    transactions = pd.DataFrame(
        {"customer_id": [0, 0, 1, 2, 2], "article_id": [0, 1, 2, 3, 3]}
    )

    assert get_purchases(transactions).equals(
        pd.DataFrame({"customer_id": [0, 1, 2], "purchases": [{0, 1}, {2}, {3}]})
    )


def test_get_predictions():
    candidates = pd.DataFrame({"customer_id": [0, 0, 1], "article_id": [0, 1, 2]})
    features = pd.DataFrame({"f1": [0, 1, 2], "f2": [0, 1, 2]})

    class MockRanker:
        def predict(self, features):
            return (features.f1 + features.f2) / 4

    ranker = MockRanker()

    assert get_predictions(candidates, features, ranker).equals(
        pd.DataFrame({"customer_id": [0, 1], "prediction": [[1, 0], [2]]})
    )


def test_fill_missing_predictions():
    predictions = pd.DataFrame({"customer_id": [0, 1], "prediction": [[1, 0], [2]]})

    assert fill_missing_predictions(predictions, [0, 1, 2], [3, 4]).equals(
        pd.DataFrame(
            {
                "customer_id": [0, 1, 2],
                "prediction": [[1, 0], [2], [3, 4]],
            }
        )
    )


def test_mean_average_precision():
    purchases = pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 3, 4],
            "purchases": [{0, 1}, {2}, {0, 3, 4}, {1, 2, 3}, {1, 2}],
        }
    )
    predictions = pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 3, 5],
            "prediction": [[1, 9], [2], [5, 6], [9, 3, 1], [1]],
        }
    )

    #  customer | prediction | purchased | average precision
    # ----------+------------+-----------+----------------------
    #  0        | 1, 9       | 0, 1      | (1 + 0) / 2   = 1/2
    #  1        | 2          | 2         | (1) / 1       = 1
    #  2        | 5, 6       | 0, 3, 4   | (0 + 0) / 2   = 0
    #  3        | 9, 3, 1    | 1, 2, 3   | (0 + 1/2) / 2 = 1/4
    #  4        | -          | 1, 2      | 0
    #  5        | 1          | -         | -
    # ----------+------------+-----------+----------------------
    #
    # MAP@2 = (1/2 + 1 + 0 + 1/4 + 0) / 5

    assert mean_average_precision(predictions, purchases, k=2) == pytest.approx(
        (1 / 2 + 1 + 0 + 1 / 4 + 0) / 5
    )
