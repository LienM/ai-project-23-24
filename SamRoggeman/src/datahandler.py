"""
    datahandler.py
    reads the processed data from the parquet files and prepares the data for the model
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle


class DataHandler:
    """
    DataHandler class

    Reads the processed data from the parquet files and prepares the data for the model by scaling and imputing the data
    """
    def __init__(self, data_folder):
        # import train_x, train_y and test_x from parquet files if needed
        train_X = pd.read_parquet(data_folder + 'train_X.parquet')
        self.train_y = pd.read_parquet(data_folder + 'train_y.parquet').values.ravel()
        test_X = pd.read_parquet(data_folder + 'test_X.parquet')
        self.validate_X = pd.read_parquet(data_folder + 'validate_X.parquet')
        self.test = pd.read_parquet(data_folder + 'test.parquet')
        self.validate = pd.read_parquet(data_folder + 'validate.parquet')
        self.validate_input_week = pd.read_parquet(data_folder + 'validate_input_week.parquet')

        # read the bestsellers_last_week from the pickle file
        with open(data_folder + 'bestsellers_last_week.pkl', 'rb') as f:
            self.bestsellers_last_week = pickle.load(f)

        # Create an imputer
        imputer = SimpleImputer(strategy='mean')

        # Fit and transform the imputer on the training data
        train_X_imputed = imputer.fit_transform(train_X)

        # Transform the test data using the same imputer
        test_X_imputed = imputer.transform(test_X)

        validate_X_imputed = imputer.transform(self.validate_X)

        # Create a scaler
        scaler = MinMaxScaler()

        # Fit and transform the training data
        self.train_X_scaled = scaler.fit_transform(train_X_imputed)

        # Transform the test data using the same scaler
        self.test_X_scaled = scaler.transform(test_X_imputed)

        # Transform the validation data using the same scaler
        self.validate_X_scaled = scaler.transform(validate_X_imputed)

if __name__ == "__main__":
    data_folder = '..\\..\\..\\Input\\Dataset\\'
    data = DataHandler(data_folder)
    # print the validation row count
    print(data.validate_X.shape)