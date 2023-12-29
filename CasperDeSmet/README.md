# Embedding Recommender

This project is an attempt at the H&M fashion recommendation challenge. We create embeddings from item data to recommend items for users based on similarity values.

## Project Structure

The `embedding-recommender` directory contains the code. In `baseline` the code from Radek's baseline can be found, adapted by Noah Daniëls. `data.py` and `device.py` are utility files for `embeddings.py`, which is responsible for creating embeddings. `ranker.py` contains all the code for the recommender itself. Finally, `evaluation.py` contains utilities for computing metrics and examples. Two jupyter notebooks are provided, `main.ipynb` and `plots.ipynb`, which make use of the previously described python files to create embeddings, make recommendations and compute metrics.

Use these notebooks to run all of the provided code.

## Data Directory Structure

In order to run most of the files provided, the data needs to be organised in the `data` directory. It should look like this.

- baseline
    - **this directory should contain the parquet files provided by Radek's baseline**
    - `articles.parquet`
    - `customers.parquet`
    - `transactions_train.parquet`
- embeddings
    - **this directory contains embeddings either created by the `embeddings.py` file or through external methods**
- images
    - **images directory provided by kaggle**
- **in the root of the data directory should be the csv files provided by kaggle**
- `articles.csv`
- `customer.csv`
- `sample_submission.csv`
- `transactions_train.csv`