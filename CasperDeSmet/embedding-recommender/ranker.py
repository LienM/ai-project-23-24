import os

import pandas
import numpy
from tqdm import tqdm
from annoy import AnnoyIndex
from sklearn.decomposition import PCA

from baseline.baseline_functions import create_submission

PERIOD = 52

# Wrapper for AnnoyIndex
class Index:
    def __init__(self, output_file, embeddings, embedding_size):
        self.embedding_size = embedding_size
        self.embedding_index = AnnoyIndex(self.embedding_size, "angular")
        self.article_index = embeddings["article_id"].to_numpy()

        # File management
        if os.path.exists(output_file):
            self.embedding_index.load(output_file)
            return

        # Build index with all embeddings
        for index, embedding in enumerate(tqdm(embeddings["embedding"].to_numpy())):
            self.embedding_index.add_item(index, embedding)
        self.embedding_index.build(5)
        self.embedding_index.save(output_file)

# Ranker that creates submission using embeddings of articles
class EmbeddingRanker:
    def __init__(self, test_week, base_path, embeddings_path, index_path, reduction_size=700):
        self.test_week = test_week
        self.embeddings = pandas.read_parquet(embeddings_path)

        # Reduce size of embeddings to speed up querying and improve embedding quality
        self.reduction = PCA(n_components=reduction_size)
        self.embeddings["embedding"] = self.reduction.fit_transform(numpy.stack(self.embeddings["embedding"].to_numpy())).tolist()

        self.index = Index(index_path, self.embeddings, reduction_size)

        self.customers = pandas.read_parquet(f"{base_path}/data/baseline/customers.parquet")["customer_id"].to_numpy()
        self.transactions = pandas.read_parquet(f"{base_path}/data/baseline/transactions_train.parquet")
        # Group transactions merged with embeddings to reduce per iteration workload
        self.grouped_transactions = (
            self.transactions.merge(self.embeddings, how="inner", on="article_id")
                             .groupby("customer_id")
        )

        # Adapted from NoahDaniels' baseline
        self.popular_data = self.transactions[self.transactions.week == self.test_week - 1] \
                                .reset_index(drop=True) \
                                .groupby("week")["article_id"].value_counts() \
                                .groupby("week").rank(method="dense", ascending=False) \
                                .groupby("week").head(25).rename('bestseller_rank').astype('int8') \
                                .to_frame().reset_index(names=["week", "article_id"])
        self.popular_data = pandas.merge(self.popular_data, self.embeddings, how="inner", on="article_id")

    # Choose correct ranking method to rank a single customer
    def get_rank_method(self, method):
        match method:
            case "plain": return self.rank_customer
            case "add_popular": return self.rank_customer_add_popular

    def rank(self, method="plain"):
        rank_method = self.get_rank_method(method)
        recommendations = [rank_method(customer_id) for customer_id in tqdm(self.customers)]
        predictions = pandas.DataFrame({"customer_id": self.customers, "prediction": recommendations})
        submission = create_submission(predictions, pandas.read_csv("data/sample_submission.csv"))
        return submission

    # Calculate weight for every article based on week article was bought in and test week
    def calculate_weights(self, weeks):
        weights = numpy.cos(2 * numpy.pi / PERIOD * (weeks - self.test_week))
        return numpy.exp(weights) / sum(numpy.exp(weights))

    def rank_customer(self, customer_id):
        customer_embedding = None
        if customer_id in self.grouped_transactions.groups:
            # retrieve all customer transactions
            transactions = self.grouped_transactions.get_group(customer_id)
            transactions = transactions[transactions.week < self.test_week]

            # calculate weights and compute weighted average
            embeddings = transactions["embedding"].to_numpy()
            weights = self.calculate_weights(transactions["week"].to_numpy())
            if embeddings.shape != (0,):
                customer_embedding = numpy.average(numpy.stack(embeddings), axis=0, weights=weights)
        if customer_embedding is None:
            customer_embedding = numpy.zeros(self.index.embedding_size)

        # Retrieve nearest neighbours using index
        nearest_neighbours = self.index.embedding_index.get_nns_by_vector(customer_embedding, 12)
        return [str(self.index.article_index[article]) for article in nearest_neighbours]

    def rank_customer_add_popular(self, customer_id):
        transactions = self.popular_data[["embedding", "week"]].head(20)
        if customer_id in self.grouped_transactions.groups:
            # retrieve all customer transactions
            transactions = pandas.concat([
                self.grouped_transactions.get_group(customer_id)[["embedding", "week"]],
                transactions
            ])
            transactions = transactions[transactions.week < self.test_week]

        # calculate weights and compute weighted average
        embeddings = transactions["embedding"].to_numpy()
        weights = self.calculate_weights(transactions["week"].to_numpy())
        if embeddings.shape != (0,):
            customer_embedding = numpy.average(numpy.stack(embeddings), axis=0, weights=weights)

        # Retrieve nearest neighbours using index
        nearest_neighbours = self.index.embedding_index.get_nns_by_vector(customer_embedding, 12)
        return [str(self.index.article_index[article]) for article in nearest_neighbours]
