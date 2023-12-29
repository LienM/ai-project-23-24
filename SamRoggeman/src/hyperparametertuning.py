from tensorflow.keras.layers import Input, Dense
from keras_tuner import RandomSearch
import tensorflow as tf

from datahandler import DataHandler

columns_to_use = ['article_id', 'product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id',
'perceived_colour_master_id', 'department_no', 'index_code',
'index_group_no', 'section_no', 'garment_group_no', 'FN', 'Active', 'has_color',
'club_member_status', 'fashion_news_frequency', 'age', 'postal_code', 'bestseller_rank', 'recency_rank', 'std_price']

# Hyperparameter tuning

# Define a function to create the ranking model using the Functional API
def build_model(hp):
    input_layer = Input(shape=(len(columns_to_use),))
    x = Dense(units=hp.Int('units', min_value=32, max_value=512, step=32))(input_layer)
    output_layer = Dense(1)(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

if __name__ == '__main__':
    # import the data
    data_folder = '..\\..\\..\\Input\\Dataset\\'
    data = DataHandler(data_folder)
    train_X_scaled = data.train_X_scaled
    train_y = data.train_y

    # define the hyperparameters
    max_trials = 10
    epochs = 10
    batch_size = 1024

    # output folder
    model_folder = '..\\..\\..\\Models\\'

    tuner = RandomSearch(build_model, objective='val_loss', max_trials=max_trials, directory='my_tuner_dir', project_name='my_ranking_model')
    tuner.search(train_X_scaled, train_y, epochs=epochs, validation_data=(train_X_scaled, train_y), verbose=1, batch_size=batch_size)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hyperparameters.values)
    best_model = build_model(best_hyperparameters)
    best_model.compile(loss='mean_squared_error', optimizer='adam')
    best_model.fit(train_X_scaled, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
    best_model.save(model_folder+'hypertuned.keras')
