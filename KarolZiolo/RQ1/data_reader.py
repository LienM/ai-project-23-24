import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from typing import Union
from scipy.sparse import (random, 
                          coo_matrix,
                          csr_matrix, 
                          csr_array,
                          vstack)
from tqdm import tqdm
import pickle
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

os.chdir("/Users/karol/Desktop/Antwerp/ai_project/")
ARTICLES_PATH = "/Users/karol/Desktop/Antwerp/ai_project/data/articles.csv"
CUSTOMER_PATH = "/Users/karol/Desktop/Antwerp/ai_project/data/customers.csv"
TRANSACTION_PATH = "/Users/karol/Desktop/Antwerp/ai_project/data/transactions_train.csv"


#######################################################################################
#                                 Data Transformations                                #
#######################################################################################
def data_preprocessing(feature_generation=False, return_encodings=False, save=False):
    customers = pd.read_csv(CUSTOMER_PATH)
    transactions = pd.read_csv(TRANSACTION_PATH)
    articles = pd.read_csv(ARTICLES_PATH)
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    

    # ARTICLE PREPROCESSING
    # article encodings
    articles = articles[['article_id'] + list(articles.select_dtypes(include=['object']).columns)]
    articles = articles.drop(columns=["detail_desc","index_code"])

    article_encodings = {}
    article_decodings = {}
    for column in articles.columns:
        names = articles[column].unique()
        encoders = np.arange(len(names))
        article_encodings[column] = dict(zip(names, encoders))
        article_decodings[column] = dict(zip(encoders, names))
        articles[column] = articles[column].apply(lambda x: article_encodings[column][x])
    # article feature selection
    cols_to_delete = ["prod_name","product_group_name","colour_group_name","perceived_colour_value_name","perceived_colour_value_name","index_group_name"]
    articles = articles.drop(columns=cols_to_delete)
    
    # CUSTOMER PREPROCESSING
    # filing NAs
    customers.FN = customers.FN.fillna(-1)
    customers.Active = customers.Active.fillna(-1)
    age_median = np.median(customers["age"].dropna())
    customers.age = customers.age.fillna(age_median)

    # customer encodings
    customer_cols = ["customer_id","club_member_status","fashion_news_frequency","postal_code"]
    customers = customers.fillna(-1)
    customer_encodings = {}
    customer_decodings = {}
    for column in customers[customer_cols]:
        names = customers[column].unique()
        if -1 in names:
            names = names[names != -1]
            encoders = np.arange(len(names))
            customer_encodings[column] = dict(zip(names, encoders))
            customer_encodings[column][-1] = -1
        else:
            encoders = np.arange(len(names))
            customer_encodings[column] = dict(zip(names, encoders))
        customer_decodings[column] = dict(zip(encoders, names))
        
        customers[column] = customers[column].apply(lambda x: customer_encodings[column][x])
    
    # TRANSACTIONS PREPROCESSING
    transactions["t_dat"] = pd.to_datetime(transactions["t_dat"])
    transactions["customer_id"] = transactions["customer_id"].apply(lambda x: customer_encodings["customer_id"][x])
    transactions["article_id"] = transactions["article_id"].apply(lambda x: article_encodings["article_id"][x])

    # FEATURE GENERATION
    if feature_generation:
        # average price
        avg_price = pd.DataFrame(transactions.groupby("customer_id")["price"].mean().rename("avg_price"), columns=["avg_price"])
        transactions = transactions.merge(avg_price, on="customer_id", how="inner") 
        
        # article selling ranking in a given month and year
        grouped_counts = transactions.groupby(["year","month","article_id"])["article_id"].count()
        articles_rank = grouped_counts.groupby(["year", "month"]).rank(ascending=False)
        articles_rank = articles_rank.rename("top_articles")
        transactions = transactions.merge(articles_rank, how="left", on=["article_id","year","month"])

        # discount
        transactions.sort_values(by=['article_id', 't_dat'], inplace=True)
        transactions['prev_price'] = transactions.groupby('article_id')['price'].shift(1)
        # Calculate the price differences
        transactions['price_diff'] = transactions['price'] - transactions['prev_price']

        transactions = transactions.drop(columns=["prev_price", "year", "month"])
        transactions["price_diff"] = transactions["price_diff"].fillna(0)
        transactions["price_diff"] = transactions["price_diff"].apply(lambda x: 1 if x < 0 else 0)
        transactions = transactions.sort_index()
    
    if save:
        transactions.to_csv("data/preprocessed/transactions.csv", index=False)
        articles.to_csv("data/preprocessed/articles.csv", index=False)
        customers.to_csv("data/preprocessed/customers.csv", index=False)

        with open("data/preprocessed/articles_encoding.pickle", "wb") as pickle_file:
            pickle.dump(article_encodings, pickle_file)
        
        with open("data/preprocessed/customers_encoding.pickle", "wb") as pickle_file:
            pickle.dump(customer_encodings, pickle_file)

        with open("data/preprocessed/articles_decoding.pickle", "wb") as pickle_file:
            pickle.dump(article_decodings, pickle_file)
        
        with open("data/preprocessed/customers_decoding.pickle", "wb") as pickle_file:
            pickle.dump(customer_decodings, pickle_file)


    if return_encodings:
        return transactions, articles, customers, article_encodings, customer_encodings, article_decodings, customer_decodings
    else:
        return transactions, articles, customers

def customer_buckets(transactions, train_test=True):
    # last purchase for customer
    if train_test:
        customer_last_purchase = transactions.groupby('customer_id')['t_dat'].max()
        merged = transactions.merge(customer_last_purchase, on='customer_id', suffixes=('', '_last_purchase'))
        # filter train and test dataset
        train_transactions = merged[merged['t_dat'] != merged['t_dat_last_purchase']]
        test_transactions = merged[merged['t_dat'] == merged['t_dat_last_purchase']]

        # get baskets
        train_buckets = train_transactions.groupby("customer_id")["article_id"].apply(list).to_dict()
        test_buckets = test_transactions.groupby("customer_id")["article_id"].apply(list).to_dict()

        return train_buckets, test_buckets
    else:
        customer_buckets = transactions.groupby("customer_id")["article_id"].apply(list).to_dict()
        return customer_buckets

def split_transactions(transactions):
    customer_last_purchase = transactions.groupby('customer_id')['t_dat'].max()
    merged = transactions.merge(customer_last_purchase, on='customer_id', suffixes=('', '_last_purchase'))
    # filter train and test dataset
    x_transactions = merged[merged['t_dat'] != merged['t_dat_last_purchase']]
    y_transactions = merged[merged['t_dat'] == merged['t_dat_last_purchase']]
    return x_transactions, y_transactions

def matrix_representation(transactions, train_test=True):
    customer_size = np.max(transactions['customer_id'])+1
    article_size = 105542
    if train_test:
        # filter train and test dataset
        x_transactions, y_transactions = split_transactions(transactions)

        # Get X matrix
        # Create the data array filled with ones
        data = np.ones_like(x_transactions.index)

        # Create the CSR matrix directly
        # assume that we investigate the purchase history therefore some articles were bought multiple times
        x_matrix = csr_matrix((data, 
                               (np.array(x_transactions['customer_id']), np.array(x_transactions['article_id']))), 
                               shape=(customer_size, article_size))

        # Get Y matrix
        # Create the data array filled with ones
        data = np.ones_like(y_transactions.index)

        # Create the CSR matrix directly
        y_matrix = csr_matrix((data, 
                               (np.array(y_transactions['customer_id']), np.array(y_transactions['article_id']))), 
                                shape=(customer_size, article_size))
        # as an output we are interested if the article was bought not its amount
        y_matrix[y_matrix>1]=1
        return x_matrix, y_matrix
    else:
        # Get test matrix
        # Create the data array filled with ones
        data = np.ones_like(transactions.index)

        # Create the CSR matrix directly
        matrix = csr_matrix((data, 
                            (np.array(transactions['customer_id']), np.array(transactions['article_id']))), 
                            shape=(customer_size, article_size))

        return matrix

def create_random_candidates(transactions, save_dir=None, num_sample=30_000_000):
    # get unique customers and articles
    unique_customers = transactions['customer_id'].unique()
    unique_articles = transactions['article_id'].unique()
    # select random customers and articles
    random_cust = np.random.choice(unique_customers, num_sample)
    random_articles = np.random.choice(unique_articles, num_sample)
    # get negative candidates dataframe
    negative_samples_df = pd.DataFrame(zip(random_cust, random_articles), columns=["customer_id","article_id",])
    # delete duplicates from original dataset
    unique_pairs = set(zip(transactions['customer_id'], transactions['article_id']))
    filtered_df = negative_samples_df[~negative_samples_df.apply(lambda row: (row['customer_id'], row['article_id']) in unique_pairs, axis=1)].copy()
    # set purchased variable
    filtered_df["purchased"] = np.zeros(len(filtered_df))
    transactions["purchased"] = np.ones(len(transactions))
    # merge dataframes
    merge = pd.concat([transactions[["customer_id","article_id", "purchased"]],filtered_df[["customer_id","article_id", "purchased"]]])
    # return shuffled dataframe
    shuffled_df = merge.sample(frac=1).reset_index(drop=True)
    if save_dir != None:
        shuffled_df.to_csv(save_dir)
    return shuffled_df

#######################################################################################
#                                    Dataset Classes                                  #
#######################################################################################

class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, data:Union[np.ndarray, coo_matrix, csr_matrix], 
                 targets:Union[np.ndarray, coo_matrix, csr_matrix], 
                 transform:bool = None):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
        self.transform = transform # Can be removed

    def __getitem__(self, index:int):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

class DatasetMF(Dataset):
    def __init__(self,trans:pd.DataFrame, transform:bool = None):
        self.transactions = trans

    def __getitem__(self, index:int):
        article_id = self.transactions["article_id"][index]
        customer_id = self.transactions["customer_id"][index]
        target = self.transactions["purchased"][index]
        return article_id, customer_id, target

    def __len__(self):
        return self.transactions.shape[0]

class SingleDataset(Dataset):
    def __init__(self, df:csr_matrix, transform:bool = None):
        self.df = df

    def __getitem__(self, index:int):
        return self.df[index]

    def __len__(self):
        return self.df.shape[0]
    
#######################################################################################
#                                Functions for Dataloader                             #
#######################################################################################

def sparse_coo_to_tensor(coo:coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)
    shape = coo.shape

    i = torch.LongTensor(np.array(indices))
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse_coo_tensor(i, v, s)
    
def sparse_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch, targets_batch = zip(*batch)
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch

def sparse_batch_collate_single(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    data_batch = batch
    if type(data_batch[0]) == csr_matrix:
        data_batch = vstack(data_batch).tocoo()
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)
    return data_batch
    
def MF_batch_collate(batch:list): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    articles_batch, customer_batch, targets_batch = zip(*batch)
    if type(articles_batch[0]) == csr_matrix:
        data_barticles_batchatch = vstack(articles_batch).tocoo()
        articles_batch = sparse_coo_to_tensor(articles_batch)
    else:
        articles_batch = torch.FloatTensor(articles_batch)
    
    if type(customer_batch[0]) == csr_matrix:
        customer_batch = vstack(customer_batch).tocoo()
        customer_batch = sparse_coo_to_tensor(customer_batch)
    else:
        customer_batch = torch.FloatTensor(customer_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = vstack(targets_batch).tocoo()
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return articles_batch, customer_batch, targets_batch

#######################################################################################
#                                      Data Loaders                                   #
#######################################################################################

def load_data(transactions, train_test=True, batch_size=1000):
    if train_test:
        # matrix representation
        x_matrix, y_matrix = matrix_representation(transactions, train_test=train_test)
        # sparse dataset
        dataset = SparseDataset(x_matrix, y_matrix)
        # split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset,[train_size, val_size])
        # load data
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=sparse_batch_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=sparse_batch_collate)
        return train_dataloader, val_dataloader
    else:
        # matrix representation
        matrix = matrix_representation(transactions, train_test=train_test)
        # sparse dataset
        dataset = SingleDataset(matrix)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=sparse_batch_collate_single)
        return dataloader

def load_data_mf(trans:pd.DataFrame, batch_size=1000):
    test_fraction = 0.1
    unique_customers = trans['customer_id'].unique()
    train_customers, test_customers = train_test_split(unique_customers, test_size=test_fraction, random_state=42)
    train_transactions = trans[trans['customer_id'].isin(train_customers)].reset_index(drop=True)
    val_transactions = trans[trans['customer_id'].isin(test_customers)].reset_index(drop=True)

    # load data
    train_dataset = DatasetMF(train_transactions)
    val_dataset = DatasetMF(val_transactions)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=MF_batch_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=MF_batch_collate)
    return train_dataloader, val_dataloader, test_customers

def load_customers_articles(customers, articles, test_customers=[], batch_size=1000):
    if len(test_customers)!=0:
        customers = customers[test_customers]
    dataset_cust = SingleDataset(customers)
    dataset_art = SingleDataset(articles)
    dataloader_cust = DataLoader(dataset_cust, batch_size=batch_size, collate_fn=sparse_batch_collate_single)
    dataloader_art = DataLoader(dataset_art, batch_size=batch_size, collate_fn=sparse_batch_collate_single)
    return dataloader_cust, dataloader_art
