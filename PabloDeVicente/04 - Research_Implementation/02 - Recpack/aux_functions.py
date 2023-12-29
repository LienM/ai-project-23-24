#dead code all of it

import numpy as np

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
            #  print("User : " + str(user))
            #info sobre las recomendaciones de un usuario
            client_row = csr.getrow(user)
            # print(client_row)
            #indice del item con maxima recomendacion 
            rec_value_index = np.argmax(client_row.data)
            # print(rec_value_index)
            #valor asociado a dicha recomendacion
            rec_value = client_row.max()
            # print("Max recommendation value : " + str(rec_value))
            #id del articulo recomendado
            article_id_rec= client_row.indices[rec_value_index]
            #  print("Recommended article id : " + str(article_id_rec))

            user_rec.add_rec(user,article_id_rec,rec_value)
        
        return user_rec
    
    def get_user_data_length(self):
        return len(self.user_data)
    

import pandas as pd
import tensorflow as tf

def shorten_transactions(original_df,x):
    
    #choose the last day in the new dataset
    last_date = pd.to_datetime('2020-09-15')
    #go back in time x months from it
    x_months_ago = last_date - pd.DateOffset(months=x)
    #it may be necessary to change datatype
    original_df['t_dat'] = pd.to_datetime(original_df['t_dat'])
    # Create a filter condition to select rows within the desired time frame.
    filter_condition = (original_df['t_dat'] >= x_months_ago) & (original_df['t_dat'] <= last_date)
    #apply the filter
    filtered_df = original_df.loc[filter_condition]

    return filtered_df

def parse_csv_line_t(line):
    data_types = [
        tf.constant('', dtype=tf.string),    # t_dat
        tf.constant('', dtype=tf.string),    # customer_id
        tf.constant(0.0, dtype=tf.float32),    # article_id
        tf.constant(0.0, dtype=tf.float32),  # price 
        tf.constant(0, dtype=tf.int32)       # sales_channel_id 
    ]

    column_names = ["t_dat", "customer_id", "article_id", "price", "sales_channel_id"]

    fields = tf.io.decode_csv(line, data_types)  
    features = dict(zip(column_names, fields))
    return features

def parse_csv_line_a(line):
    data_types = [
        tf.constant('', dtype=tf.string),        # article_id
        tf.constant('', dtype=tf.string),        # product_code
        tf.constant('', dtype=tf.string),        # prod_name
        tf.constant(0, dtype=tf.int32),          # product_type_no
        tf.constant('', dtype=tf.string),        # product_type_name
        tf.constant('', dtype=tf.string),        # product_group_name
        tf.constant(0, dtype=tf.int32),          # graphical_appearance_no
        tf.constant('', dtype=tf.string),        # graphical_appearance_name
        tf.constant(0, dtype=tf.int32),          # colour_group_code
        tf.constant('', dtype=tf.string),        # colour_group_name
        tf.constant(0, dtype=tf.int32),          # perceived_colour_value_id
        tf.constant('', dtype=tf.string),        # perceived_colour_value_name
        tf.constant(0, dtype=tf.int32),          # perceived_colour_master_id
        tf.constant('', dtype=tf.string),        # perceived_colour_master_name
        tf.constant(0, dtype=tf.int32),          # department_no
        tf.constant('', dtype=tf.string),        # department_name
        tf.constant('', dtype=tf.string),        # index_code
        tf.constant('', dtype=tf.string),        # index_name
        tf.constant(0, dtype=tf.int32),          # index_group_no
        tf.constant('', dtype=tf.string),        # index_group_name
        tf.constant(0, dtype=tf.int32),          # section_no
        tf.constant('', dtype=tf.string),        # section_name
        tf.constant(0, dtype=tf.int32),          # garment_group_no
        tf.constant('', dtype=tf.string),        # garment_group_name
        tf.constant('', dtype=tf.string)         # detail_desc
    ]
    column_names = [
        "article_id",
        "product_code",
        "prod_name",
        "product_type_no",
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "perceived_colour_value_id",
        "perceived_colour_value_name",
        "perceived_colour_master_id",
        "perceived_colour_master_name",
        "department_no",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "section_no",
        "section_name",
        "garment_group_no",
        "garment_group_name",
        "detail_desc"
    ]

    fields = tf.io.decode_csv(line, data_types)  
    features = dict(zip(column_names, fields))
    return features

##cut short method for MapObject (does not work)
def cutShort(item, months):
    database, date_column = item  # Unpack the item into your database and date_column

    # We exclude the last week
    target_date = pd.to_datetime('2020-09-15')
    # And count x months back
    x_months_ago = target_date - pd.DateOffset(months=months)

    # Change date_column to datetime
    database[date_column] = pd.to_datetime(database[date_column])

    # Create a filter condition to select rows within the desired time frame.
    filter_condition = (database[date_column] >= x_months_ago) & (database[date_column] <= target_date)
    filtered_df = database[filter_condition]

    return filtered_df