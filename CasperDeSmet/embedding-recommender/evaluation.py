import json
import numpy
import pandas

from sklearn.metrics.pairwise import cosine_similarity

from ranker import EmbeddingRanker
from baseline.baseline_functions import customer_hex_id_to_int, get_purchases, mean_average_precision

# compute competition score for given predictions and week
def evaluate(base_path, predictions, test_week=104):
    predictions = pandas.read_csv(f"{base_path}/{predictions}")
    transactions = pandas.read_parquet(f"{base_path}/data/baseline/transactions_train.parquet")
    predictions["customer_id"] = customer_hex_id_to_int(predictions["customer_id"])
    predictions["prediction"] = predictions["prediction"].apply(lambda prediction: [int(article_id) for article_id in prediction.split(" ")])
    purchases = get_purchases(transactions[transactions.week == test_week])
    return mean_average_precision(predictions, purchases, 12)

# extract predictions from csv file and transform into transactional dataframe
def get_predictions(base_path, predictions):
    predictions = pandas.read_csv(f"{base_path}/{predictions}")
    predictions["customer_id"] = customer_hex_id_to_int(predictions["customer_id"])
    predictions["prediction"] = predictions["prediction"].apply(lambda prediction: [int(article_id) for article_id in prediction.split(" ")])
    predictions = predictions.explode("prediction").rename(columns={"prediction": "article_id"})
    return predictions

# compute recall for given predictions and week
def recall(base_path, predictions, test_week=104):
    predictions = get_predictions(base_path, predictions)

    # courtesy of Noah Daniëls, who gracefully allowed me to use this
    transactions = pandas.read_parquet(f"{base_path}/data/baseline/transactions_train.parquet")
    purchases = transactions[transactions.week == test_week][["customer_id", "article_id"]].drop_duplicates()

    joined = pandas.merge(purchases, predictions, how="inner").drop_duplicates()
    relevant_selected = joined.groupby("customer_id").count()
    relevant_total = purchases.groupby("customer_id").count()

    recall = relevant_selected.divide(relevant_total, fill_value=0)
    return recall.mean().values[0]

# compute average bestseller rank for given predictions and week
def average_bestseller_rank(base_path, predictions, test_week=104):
    # get ranked list of popular items for test week
    transactions = pandas.read_parquet(f"{base_path}/data/baseline/transactions_train.parquet")
    popular_data = transactions[transactions.week == test_week - 1] \
                .reset_index(drop=True)["article_id"].value_counts() \
                .rank(method="dense", ascending=False) \
                .rename("bestseller_rank").astype('int32') \
                .to_frame().reset_index(names=["article_id"])

    # compare predictions to popular items
    predictions = get_predictions(base_path, predictions)
    besteller_rank = pandas.merge(predictions, popular_data, how="inner")
    return besteller_rank["bestseller_rank"].mean()

# apply plain text template to text data of article
def plain(base_path, article_id):
    articles = pandas.read_csv(f"{base_path}/data/articles.csv").fillna("")
    with open(f"{base_path}/embedding-recommender/text_embeddings.json", "r") as config_file:
        config = json.load(config_file)
        columns = config["columns"]
        templates = config["templates"]
    article, = articles.loc[articles["article_id"] == article_id][columns].to_numpy()
    article_text = templates["plain"].format(**{column: field for column, field in zip(columns, article)})
    return article_text

# find most similar item to given item from given embeddings
def most_similar(article_id, embeddings):
    article, = embeddings.loc[embeddings["article_id"] == article_id]["embedding"].to_numpy()
    all_articles = numpy.stack(embeddings.loc[embeddings["article_id"] != article_id]["embedding"].to_numpy(), axis=0)
    similarities = cosine_similarity(all_articles, article.reshape(1, -1))
    most_similar_article = embeddings.loc[embeddings["article_id"] != article_id].to_numpy()[numpy.argmax(similarities)][0]
    return most_similar_article

# worked out example of finding most similar item to given item
def example_similarity(base_path, article_id):
    print("article text:", plain(base_path, article_id))

    text = pandas.read_parquet(f"{base_path}/data/embeddings/text_plain_average_embeddings.parquet")
    image = pandas.read_parquet(f"{base_path}/data/embeddings/image_embeddings.parquet")

    most_similar_article_text = most_similar(article_id, text)
    print("most similar article on text embeddings:", most_similar_article_text)
    print("most similar article text:", plain(base_path, most_similar_article_text))

    most_similar_article_image = most_similar(article_id, image)
    print("most similar article on image embeddings:", most_similar_article_image)

# worked out example of finding most similar item to user profile embedding of given user
def example_customer(base_path, customer_id, embedding_type="text_plain_average", test_week=105, reduction_size=700):
    embeddings = f"{base_path}/data/embeddings/{embedding_type}_embeddings.parquet"
    index = f"{base_path}/data/indices/{embedding_type}_index_{reduction_size}.ann"
    ranker = EmbeddingRanker(test_week, base_path, embeddings, index, reduction_size=reduction_size)

    user_profile = ranker.grouped_transactions.get_group(customer_id)["article_id"].to_numpy()
    recommendations = ranker.rank_customer(customer_id)
    print(f"{user_profile=}\n{recommendations=}")
    print("user profile article texts:", [plain(base_path, article_id) for article_id in user_profile])
    print("recommendations article texts:", [plain(base_path, int(article_id)) for article_id in recommendations[:3]])
