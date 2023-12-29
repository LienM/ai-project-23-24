"""
    File: generate_predictions.py
    Generates predictions for the test set using the best provided model
"""

import pandas as pd
import tensorflow as tf

from datahandler import DataHandler
from Radek.utils import customer_hex_id_to_int

batch_size = 1024


def make_predictions(data_handler, model_folder, model_name, submission_file, output_folder):

    # load the provided model
    model = tf.keras.models.load_model(f'{model_folder}\\{model_name}.keras')

    # Predict the probability of a purchase using the best model such that the model can be used for ranking
    data_handler.test['preds'] = model.predict(data_handler.test_X_scaled, batch_size=batch_size, verbose=1)

    preds = []
    # sort the predictions by customer_id and then by the probability of a purchase, from high to low chance
    # group the predictions by customer_id and take article_id column
    c_id2predicted_article_ids = data_handler.test \
        .sort_values(['customer_id', 'preds'], ascending=False) \
        .groupby('customer_id')['article_id'].apply(list).to_dict()

    # create the submission file
    sub = pd.read_csv(submission_file)

    # Get the predicted article_ids augmented with the bestsellers_last_week to get 12 predictions
    for c_id in customer_hex_id_to_int(sub.customer_id):
        pred = c_id2predicted_article_ids.get(c_id, [])
        pred = pred + data_handler.bestsellers_last_week
        preds.append(pred[:12])

    # convert the article_ids back to strings
    preds = [' '.join(['0' + str(p) for p in ps]) for ps in preds]
    sub.prediction = preds
    sub.to_csv( f"{output_folder}{model_name}.csv", index=False)


if __name__ == "__main__":
    model_folder = '..\\..\\..\\Models\\'
    data_folder = '..\\..\\..\\Input\\Dataset\\'
    submission_file = '..\\..\\..\\Input\\Dataset\\sample_submission.csv'
    output_folder = f'..\\..\\..\\Output\\'
    data_handler = DataHandler(data_folder)

    # make predictions using some notable models

    # make_predictions("ranking_model_13_layers_192_hidden_size")
    # make_predictions("ranking_model_9_layers_320_hidden_size")
    make_predictions(data_handler, model_folder, "ranking_model_1_layers_64_hidden_size", submission_file, output_folder)
