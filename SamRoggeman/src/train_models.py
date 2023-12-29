"""
file: train_models.py
This file contains the code to train the models and find the best model
"""

import csv
import os
import shutil
import matplotlib.pyplot as plt
from datahandler import DataHandler
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from keras_tuner.tuners import RandomSearch

from Radek.average_precision import calculate_apk_dicts
batch_size = 1024*50
epochs = 10
data_folder = '..\\..\\..\\Input\\Dataset\\'
model_folder = '..\\..\\..\\Models\\'
columns_to_use = ['article_id', 'product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id',
'perceived_colour_master_id', 'department_no', 'index_code',
'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active', 'has_color',
'club_member_status', 'fashion_news_frequency', 'age', 'postal_code', 'bestseller_rank', 'recency_rank', 'std_price']


# Define your neural network model for ranking
def create_ranking_model(input_dim, nr_hidden_layers=1, hidden_size=64, dropout=0.2,name=None):
    """
    Create a ranking model using the Functional API
    :param input_dim:  the number of features
    :param nr_hidden_layers:  the number of hidden layers
    :param hidden_size:  the number of neurons per hidden layer
    :param dropout:  the dropout rate
    :param name: the name of the model
    :return:  the ranking model
    """
    # if no name is given, create a name
    if not name:
        name = f'ranking_model_{nr_hidden_layers}_layers_{hidden_size}_hidden_size'
    model_input = Input(shape=(input_dim,))
    # add first hidden layer to the input layer
    dense_layer = Dense(hidden_size, activation='relu')(model_input)

    # add the remaining hidden layers
    for _ in range(nr_hidden_layers - 1):
        dense_layer = Dense(hidden_size, activation='relu')(dense_layer)

    # add the output layer
    output_layer = Dense(1, activation='linear')(dense_layer)

    # create the model
    model = tf.keras.Model(inputs=model_input, outputs=output_layer,
                           name=name)
    return model



def train_models(train_X_scaled, train_y):
    """
    Train a ranking model for each combination of the number of hidden layers and the number of neurons per layer
    :param train_X_scaled:  the scaled training set
    :param train_y:  the training labels
    """
    # min and max values for the number of hidden layers and the number of neurons per layer
    min_layers = 1
    max_layers = 16

    min_hidden_size = 64
    max_hidden_size = 512

    # step size for the number of hidden layers and the number of neurons per layer
    layer_step = 4
    hidden_size_step = 64

    # calculate the number of models to train
    nr_models = ((max_layers - min_layers) // layer_step + 1) * ((max_hidden_size - min_hidden_size) // hidden_size_step + 1)
    print(f'nr_models: {nr_models}')
    nr = 0
    # train a model for each combination of the number of hidden layers and the number of neurons per layer
    for nr_hidden_layers in range(min_layers, max_layers + 1, layer_step):
        # print the progress
        print(f"nr of hidden layers: {nr_hidden_layers}")
        for hidden_size in range(min_hidden_size, max_hidden_size + 1, hidden_size_step):
            modelname = f'ranking_model_{nr_hidden_layers}_layers_{hidden_size}_hidden_size'
            nr+=1
            # if the model is already trained, skip it
            if os.path.isfile(f'{model_folder}/{modelname}.keras'):
                continue
            # create the model and compile it
            model = create_ranking_model(input_dim=len(columns_to_use), nr_hidden_layers=nr_hidden_layers, hidden_size=hidden_size, name= modelname)
            model.compile(loss='mean_squared_error', optimizer='adam')
            # train without any terminal output
            model.fit(train_X_scaled, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
            # save the model
            model.save(f'{model_folder}/{model.name}.keras')



def find_best_model(validate_X_scaled, validate_y, test, validate_input_week):
    """
    Find the best model by using the validation set
    :param validate_X_scaled:  the scaled validation set
    :param validate_y:  the validation labels
    :param test:  the test set (true purchases)
    :param validate_input_week: the validation input week to augment the predictions with the repuchases
    :return:
    """
    # load all models
    models = []
    for file in os.listdir(model_folder):
        if file.endswith('.keras'):
            models.append(tf.keras.models.load_model(f'{model_folder}/{file}'))

    # find the best model
    best_model = None
    best_score = 0
    # calculate the average precision at k by using the validation set
    test_week_purchases_by_cust_validation = test.groupby('customer_id')['article_id'].apply(list).to_dict()
    mapks = {}

    for model in models:
        column_name = f'preds_{model.name}'
        # predict the items purchased in the test week
        validate_input_week[column_name] = model.predict(validate_X_scaled)
        # calculate the average precision at k for the test week
        c_id2predicted_article_ids_validation = validate_input_week \
            .sort_values(['customer_id', column_name], ascending=False) \
            .groupby('customer_id')['article_id'].apply(list).to_dict()

        apk_val = calculate_apk_dicts(dict_of_gts=test_week_purchases_by_cust_validation,
                                      dict_of_preds=c_id2predicted_article_ids_validation)
        print(f'apk for {model.name}: {apk_val}')
        nr_hidden_layers = int(model.name.split('_')[2])
        hidden_size = int(model.name.split('_')[4])
        if nr_hidden_layers not in mapks:
            mapks[nr_hidden_layers] = {}
        mapks[nr_hidden_layers][hidden_size] = apk_val

        if apk_val > best_score:
            best_model = model
            best_score = apk_val

    return best_model, mapks
def makeplotMapK(mapks):
    """
    make a plot of the mapks, with the number of neurons per layer on the x-axis and the mapk on the y-axis and number of hidden layers as series
    :param mapks:  dict of dicts, where the first key is the number of hidden layers,
        the second key is the number of neurons per layer and the value is the mapk
    """
    for nr_hidden_layers, series in mapks.items():
        # sort the series by key
        series = {k: v for k, v in sorted(series.items(), key=lambda item: item[0])}
        hidden_size = list(series.keys())
        apk_val = list(series.values())


        plt.plot(hidden_size, apk_val, label=f'{nr_hidden_layers} hidden layers')
    plt.legend()
    plt.title('Average Precision at K(=12)')
    plt.xlabel('Number of neurons/layer')
    plt.ylabel('Average Precision at K(=12)')
    plt.savefig('./output/hidden_layers.png')
    plt.show()

def modelSummary():
    """
    Print the summary of all models
    :return:
    """
    # load all models
    models = []
    for file in os.listdir(model_folder):
        if file.endswith('.keras'):
            models.append(tf.keras.models.load_model(f'{model_folder}/{file}'))
    for model in models:
        model.summary()



def handle_model_training(data: DataHandler, model_folder):
    """
    Train the models and find the best model
    :param data:  the datahandler
    :param model_folder:  the folder where the models are stored
    :return: the mapks for each model
    """
    if not os.path.isdir(model_folder):
        # create model folder
        os.mkdir(model_folder)
    # train the models using the training set
    train_models(data.train_X_scaled, data.train_y)
    # find the best model using the validation set and calculate the average precision at k for the test set
    best_model, mapks = find_best_model(data.validate_X_scaled, data.validate, data.test,data.validate_input_week)
    best_model.summary()
    # mapks = {13: {128: 0.512279816134107, 192: 0.517068718408337, 256: 0.5002576464657217, 320: 0.44673369547730407, 384: 0.4601087987665479, 448: 0.4994916106728824, 64: 0.4877935586323192}, 1: {128: 0.42971012635560824, 192: 0.5002863239782738, 256: 0.45008219219811724, 320: 0.5185888655969477, 384: 0.4453363177026598, 448: 0.44689232323938527, 512: 0.4560267679300869, 64: 0.40006891655777155}, 5: {128: 0.4896824998337128, 192: 0.4856015713354663, 256: 0.5058273052734822, 320: 0.4936424676057455, 384: 0.49949202362274464, 448: 0.494659991833011, 512: 0.5238538327174596, 64: 0.5289485862850578}, 9: {128: 0.5203026662736008, 192: 0.5200065227068728, 256: 0.5262727719862984, 320: 0.5340056705430928, 384: 0.5211481016401629, 448: 0.5197821488245133, 512: 0.47103674615327545, 64: 0.4959754584538742}}
    makeplotMapK(mapks)
    return mapks


if __name__ == "__main__":
    data = DataHandler(data_folder)

    mapks = handle_model_training(data, model_folder)
