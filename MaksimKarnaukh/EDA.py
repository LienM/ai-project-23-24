import pandas as pd
import seaborn as sns

# Individual columns

# Combinations of columns

if __name__ == '__main__':

    df = pd.read_csv('datasets/articles.csv')

    # Display the first few rows of the dataset
    print(df.head())
    # Get a summary of the dataset's structure
    print(df.info())
    # Summary statistics of numerical columns
    print(df.describe())
    # Number of missing values in each column
    print(df.isnull().sum())