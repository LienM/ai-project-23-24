##IN THEORY THIS FILE IS THE SAME AS RECPACK4. 
##BUT WHEN TESTING AND DEBUGGING RECPACK4 IS THE FILE USED, SO THIS ONE MAY BE OUTDATED 
##(it is just so that it makes it easier calling the framewrok from RECPACK5)

import tqdm as notebook_tqdm
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from aux_functions import *
from PipelineBuilder_modified import * 


# from recpack.pipelines import PipelineBuilder
from PipelineBuilder_modified import * 
from recpack.scenarios import WeakGeneralization, Timed
from recpack.preprocessing.preprocessors import DataFramePreprocessor
from recpack.preprocessing.filters import MinItemsPerUser, MinUsersPerItem

# Dataframe setup
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4)
# Seaborn setup
sns.set_theme(style="whitegrid")
sns.set_palette("pastel")

def pipeline_mod():
    #1:  Data collection
    transactions_path = '../../00 - Data/transactions/transactions_train.csv'
    transactions = pd.read_csv(transactions_path)
    print("Original data has size of : " + str(len(transactions)))

    #transform datetime to unix epoch
    transactions['timestamp'] = pd.to_datetime(transactions['t_dat']).astype(int) / 10**9
    transactions.drop(columns=['t_dat'], inplace=True)

    sample = 0.5
    #0.5 % with 15894162 records MAX
    transactions_sample = transactions.sample(frac=sample, random_state=40)
    print("Created a sample of " + str(sample) + " % with " + str(len(transactions_sample)) + " records")

    #2: Data preprocessing

    #        item1    item2   item3
    #usr1      x                x
    #usr2       x       x
    proc = DataFramePreprocessor(item_ix='article_id', user_ix='customer_id', timestamp_ix='timestamp')
    # #every user has at least 2 items bought
    proc.add_filter(MinUsersPerItem(2, item_ix='article_id', user_ix='customer_id'))
    # #every item is bought at least twice
    proc.add_filter(MinItemsPerUser(2, item_ix='article_id', user_ix='customer_id'))

    interaction_matrix = proc.process(transactions)

    from datetime import datetime
    #As i really dont have validation data (it is hidden by kaggle, i set both values to same date). 'delta_in' before 't' will be used for training and 'delta_out' weeks after 't' will be used for testing
    t = datetime(2020, 9, 15).timestamp()
    t_validation = datetime(2020, 9, 14).timestamp()
    #maybe 9?
    delta_in = 10 * 604800
    # 1 semana = 604800
    delta_out = 604800

    #3 : Create scenario
    scenario = Timed(t, t_validation=t_validation, validation=True, delta_in = delta_in, delta_out = delta_out, seed =1)
    # scenario = Timed() 
    scenario.split(interaction_matrix)

    #4 : Create the builder object
    builder = PipelineBuilder()
    builder.set_data_from_scenario(scenario)

    #adds algorithms to use later on. Baseline algorithim, just recommends popular stuff
    # builder.add_algorithm('Popularity') 
    builder.add_algorithm('ItemKNN', grid={
        'K': [100, 200, 500],
        'similarity': ['cosine', 'conditional_probability'],
    })
    #Set the metric for optimisation of parameters in algorithms. What is NDCGK ??
    builder.set_optimisation_metric('NDCGK', K=10)

    #adds metric for evaluation
    #NDCGK = Normalized Discounted Cumulative Gain at K
    builder.add_metric('NDCGK', K=[10, 20, 50])
    builder.add_metric('CoverageK', K=[10, 20])

    # #5 : Create and run the pipeline
    # pipeline = builder.build()
    # pipeline.run()
    # # x_preds = pipeline.run2()

    # #6 : Get results
    # pipeline.get_metrics()
    # # pipeline.optimisation_results
    # #pipeline.saveResults()

    pipeline = builder.build()
    csr = pipeline.run2()
    user_rec = UserRecommendations.fill_user_rec(csr)

    #ITEM BEING RECOMMENDED
    # user_rec.get_rec_user(6)[0][0]
    article_ids_array = user_rec.get_article_ids()
    decoded_items = []
    #i get back the ids of the articles
    item_id_mapping = proc.item_id_mapping.set_index(interaction_matrix.ITEM_IX)[proc.item_ix].to_dict()
    
    for item_id in item_id_mapping:
        #and i put onto a list of ids each of the decoded_ids
        decoded_item = item_id_mapping[item_id]
        decoded_items.append(decoded_item)

    #original list of articles
    articles_path = '../../00 - Data/articles/articles.csv'
    articles_df = pd.read_csv(articles_path)

    #recommended articles +  info
    recommended_items = articles_df[articles_df['article_id'].isin(decoded_items)]

    return recommended_items


#aux class that i didnt feel like having in a separate file
class UserRecommendations:
    def __init__(self):
        self.user_data = {}

    def add_rec(self, user_id, item_id, recommendation_value):
        if user_id not in self.user_data:
            self.user_data[user_id] = []
        self.user_data[user_id].append((item_id, recommendation_value))

    def get_rec_user(self, user_id):
        return self.user_data.get(user_id, [])
    
    def get_article_ids(self):
        article_ids = []
        for user_id, recs in self.user_data.items():
            for item_id, _ in recs:
                article_ids.append(item_id)
        return article_ids


    def fill_user_rec(csr):

        user_rec = UserRecommendations()
        #get the list of every user who has been recomended smth
        user_ids = set()
        for row in range(csr.shape[0]):
            if csr.indptr[row] != csr.indptr[row + 1]:
                user_ids.add(row)

        for user in user_ids:
            print("User : " + str(user))
            #info sobre las recomendaciones de un usuario
            client_row = csr.getrow(user)
            # print(client_row)
            #indice del item con maxima recomendacion 
            rec_value_index = np.argmax(client_row.data)
            # print(rec_value_index)
            #valor asociado a dicha recomendacion
            rec_value = client_row.max()
            print("Max recommendation value : " + str(rec_value))
            #id del articulo recomendado
            article_id_rec= client_row.indices[rec_value_index]
            print("Recommended article id : " + str(article_id_rec))

            user_rec.add_rec(user,article_id_rec,rec_value)
        
        return user_rec
    
    def get_user_data_length(self):
        return len(self.user_data)