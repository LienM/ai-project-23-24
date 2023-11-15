import pandas as pd
import tensorflow as tf

def shorten_transactions(transactions_path,x):

    transactions = pd.read_csv(transactions_path)
    print("Number of elements in database before : " + str(len(transactions)))
    # a week prior to last date
    target_date = pd.to_datetime('2020-09-15')

    # Calculate the date 5 months before the target date.
    five_months_ago = target_date - pd.DateOffset(months=x)
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    # Create a filter condition to select rows within the desired time frame.
    filter_condition = (transactions['t_dat'] >= five_months_ago) & (transactions['t_dat'] <= target_date)
    filtered_df = transactions.loc[filter_condition]
    print("Number of elements in database after : " + str(len(filtered_df)))
    print("Database contains recrods from : " + str(filtered_df.t_dat.max()) + " to : " +str(filtered_df.t_dat.min()))
    route = "../00 - Data/transactions_train/short_transactions.csv"

    # top = 10000
    # print("Cutting it short to " + str(top))
    # filtered_df = filtered_df.head(top)
    filtered_df.to_csv(route,index = False)

    return route


def parse_csv_line_t(line):
    data_types = [
        tf.constant('', dtype=tf.string),    # t_dat
        tf.constant('', dtype=tf.string),    # customer_id
        tf.constant(0.0, dtype=tf.float32),    # article_id
        tf.constant(0.0, dtype=tf.float32),  # price (
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