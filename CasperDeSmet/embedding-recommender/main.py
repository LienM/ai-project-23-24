import time

import pandas
import numpy
from loguru import logger

from ranker import EmbeddingRanker
from baseline.baseline_functions import get_purchases, mean_average_precision, customer_hex_id_to_int
from embeddings import create_image_embeddings, create_text_embeddings, IMAGE_EMBEDDING_SIZE, TEXT_EMBEDDING_SIZE, concatenate_embeddings

def rank():
    test_week = 105
    reduction_size = 700
    embeddings = "data/embeddings/text_embeddings_plain.parquet"
    index = f"data/indices/text_index_{reduction_size}.ann"
    ranker = EmbeddingRanker(test_week, embeddings, index, reduction_size=reduction_size)
    predictions = ranker.rank("add_popular")
    predictions.to_csv("submissions/text.csv.gz", index=False)

def evaluate():
    test_week = 104
    predictions = pandas.read_csv("submissions/baseline.csv.gz")
    transactions = pandas.read_parquet("data/baseline/transactions_train.parquet")
    predictions["customer_id"] = customer_hex_id_to_int(predictions["customer_id"])
    predictions["prediction"] = predictions["prediction"].apply(lambda prediction: [int(article_id) for article_id in prediction.split(" ")])
    purchases = get_purchases(transactions[transactions.week == test_week])
    print(mean_average_precision(predictions, purchases))

def average_bestseller():
    test_week = 105
    transactions = pandas.read_parquet("data/baseline/transactions_train.parquet")
    popular_data = transactions[transactions.week == test_week - 1] \
                .reset_index(drop=True) \
                .groupby("week")["article_id"].value_counts() \
                .groupby("week").rank(method="dense", ascending=False) \
                .groupby("week").head(25).rename('bestseller_rank').astype('int8') \
                .to_frame().reset_index(names=["week", "article_id"])

@logger.catch
def main():
    # create_image_embeddings("data/images.nosync", "data/embeddings/image_embeddings.parquet")
    # create_text_embeddings("data/articles.csv", "data/embeddings/text_embeddings_plain.parquet", template="plain")
    # concatenate_embeddings("data/embeddings/text_embeddings_plain.parquet", "data/embeddings/image_embeddings.parquet", "data/embeddings/concatenated_embeddings.parquet")

    rank()
    # evaluate()

if __name__ == "__main__":
    main()