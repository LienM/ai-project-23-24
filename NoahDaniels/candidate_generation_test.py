import pandas as pd
import pytest

from candidate_generation import *


def test_recall():
    test_data = pd.DataFrame(
        {
            "customer_id": [0, 0, 1, 2, 2, 2, 3, 3, 3, 5],
            "article_id": [0, 1, 2, 0, 3, 4, 1, 2, 5, 0],
        }
    )
    predictions = pd.DataFrame(
        {
            "customer_id": [0, 0, 1, 1, 2, 3, 4],
            "article_id": [0, 9, 2, 9, 4, 0, 9],
        }
    )

    #  customer | predicted | actual  | recall
    # ----------+-----------+---------+-------
    #  0        | 0, 9      | 0, 1    | 1/2
    #  1        | 2, 9      | 2       | 1
    #  2        | 4         | 0, 3, 4 | 1/3
    #  3        | 0         | 1, 2, 5 | 0
    #  4        | 9         | -       | -
    #  5        | -         | 0       | 0
    # ----------+-----------+---------+--------
    # total recall:  (1/2 + 1 + 1/3 + 0 + 0)/5

    assert recall(predictions, test_data) == pytest.approx(
        (1 / 2 + 1 + 1 / 3 + 0 + 0) / 5
    )


def test_add_relative_week():
    data = pd.DataFrame(
        {"customer_id": [0, 0, 0, 0, 0, 1, 1, 1], "week": [0, 2, 2, 5, 5, 1, 1, 10]}
    )

    result = add_relative_week(data)
    expected = pd.concat(
        [data, pd.Series([2, 1, 1, 0, 0, 1, 1, 0], name="week_rel")], axis=1
    )
    assert result.equals(expected)


def test_by_week():
    data = pd.DataFrame(
        {
            "customer_id": [0, 1, 0, 1, 0, 0, 0, 0, 0],
            "week": [0, 0, 0, 1, 1, 2, 2, 3, 5],
            "article_id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        }
    )

    def f(x):
        return x.sort_values("article_id").head(1)

    assert by_week([5, 4, 3], data, 2, f).equals(
        pd.DataFrame(
            {
                "customer_id": [0, 0, 1],
                "week": [5, 4, 3],
                "article_id": [7, 5, 3],
            }
        )
    )


def test_popularity():
    baskets = pd.DataFrame({"customer_id": [0, 0, 1], "week": [5, 10, 10]})
    data = pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 3, 4, 5, 6, 7],
            "week": [3, 3, 4, 4, 4, 5, 5, 9],
            "article_id": [0, 0, 1, 0, 1, 0, 0, 2],
        }
    )

    assert candidates_popularity(baskets, data, 1, 1).equals(
        pd.DataFrame(
            {
                "customer_id": [0, 0, 1],
                "week": [5, 10, 10],
                "article_id": [1, 2, 2],
                "rank": [1, 1, 1],
            }
        )
    )


def test_repurchase():
    baskets = pd.DataFrame({"customer_id": [0, 0, 1], "week": [5, 10, 10]})
    data = pd.DataFrame(
        {
            "customer_id": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "week": [3, 3, 4, 4, 4, 5, 5, 9, 9, 10],
            "article_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        }
    )

    assert candidates_repurchase(baskets, data, 1, False).equals(
        pd.DataFrame(
            {
                "customer_id": [0, 0, 0, 0, 1],
                "week": [5, 5, 5, 10, 10],
                "article_id": [2, 3, 4, 7, 8],
            }
        )
    )
