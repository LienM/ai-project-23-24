import os
import json

import torch
import numpy
import pandas
from tqdm import tqdm
from transformers import pipeline
from torchvision.models import vgg19_bn, VGG19_BN_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from data import load_data
from device import device

IMAGE_EMBEDDING_SIZE = 4096
TEXT_EMBEDDING_SIZE = 768

BATCH_SIZE = 32
CLASS_TOKEN = 0

# Create embeddings for images in a given directory and write them to a parquet file
def create_image_embeddings(base_path, image_directory, output_file):
    if os.path.exists(f"{base_path}/{output_file}"):
        return pandas.read_parquet(f"{base_path}/{output_file}")

    data = load_data(f"{base_path}/{image_directory}", BATCH_SIZE)
    columns = ["article_id", "embedding"]
    embeddings = pandas.DataFrame(columns=columns)

    # use first fc layer of classifier layer of VGG to compute embedding
    vgg = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).to(device)
    embedder = create_feature_extractor(vgg, return_nodes={"classifier.0": "embedding"})
    with torch.no_grad():
        for inputs, article_ids in tqdm(data):
            article_ids = numpy.array(article_ids).astype(numpy.int32)
            outputs = embedder(inputs.to(device))["embedding"]
            batch_embeddings = pandas.DataFrame(list(zip(article_ids, outputs.cpu().numpy())), columns=columns)
            embeddings = pandas.concat([embeddings, batch_embeddings])
    embeddings.to_parquet(f"{base_path}/{output_file}")
    return embeddings

# Create embeddings for text of all articles and write them to a parquet file
def create_text_embeddings(base_path, articles_file, output_file, template="plain", average=True):
    if os.path.exists(f"{base_path}/{output_file}"):
        return pandas.read_parquet(f"{base_path}/{output_file}")

    bert = pipeline(task="feature-extraction", model="bert-base-uncased", device=device)

    with open(f"{base_path}/embedding-recommender/text_embeddings.json", "r") as config_file:
        config = json.load(config_file)
        columns = config["columns"]
        templates = config["templates"]

    # Apply template to article text
    articles = pandas.read_csv(f"{base_path}/{articles_file}", dtype=str).fillna(value="")
    article_text = lambda article: templates[template].format(**{column: field for column, field in zip(columns, article)})
    article_texts = (article_text(article) for article in tqdm(articles[columns].to_numpy()))

    # Compute bert embedding from create article text
    pool = lambda embedding: numpy.average(embedding, axis=0) if average else embedding[CLASS_TOKEN]
    embeddings = [pool(embedding) for embedding, in bert(article_texts)]
    embeddings = pandas.DataFrame({"article_id": articles["article_id"].astype(numpy.int32), "embedding": embeddings})
    embeddings.to_parquet(f"{base_path}/{output_file}")
    return embeddings

# Concatenate image and text embeddings
def concatenate_embeddings(base_path, text_embeddings_file, image_embeddings_file, output_file):
    if os.path.exists(f"{base_path}/{output_file}"):
        return pandas.read_parquet(f"{base_path}/{output_file}")

    text_embeddings = pandas.read_parquet(f"{base_path}/{text_embeddings_file}")
    image_embeddings = pandas.read_parquet(f"{base_path}/{image_embeddings_file}")

    concatenate = lambda row: [*row.iloc[0], *row.iloc[1]]
    concatenated_embeddings = text_embeddings.merge(image_embeddings, on="article_id", suffixes=("_text", "_image"), how="inner")
    concatenated_embeddings["embedding"] = concatenated_embeddings[["embedding_text", "embedding_image"]].agg(concatenate, axis="columns")
    concatenated_embeddings.drop(["embedding_text", "embedding_image"], axis=1)
    concatenated_embeddings.to_parquet(f"{base_path}/{output_file}")
    return concatenated_embeddings
