import time
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial

# NOTE: code taken from "https://www.kaggle.com/code/julian3833/h-m-collaborative-filtering-user-user" for use in comparisons
# Was not ever executed for more than a couple thousand users due to the immensely long runtime

N_SIMILAR_USERS = 30
MINIMUM_PURCHASES = 3
START_WEEK = 102
DROP_PURCHASED_ITEMS = False
DROP_USER_FROM_HIS_NEIGHBORHOOD = False
TEST_RUN = False
TEST_SIZE = 10000



def flatten(l):
    """ Flatten a list of lists"""
    return [item for sublist in l for item in sublist]


def compare_vectors(v1, v2):
    """Compare lists of purchased product for two given users
    v1 stands for the "vector representation for user 1", which is a list of the purchases of u1

    Returns:
        A value between 0 and 1 (similarity)
    """
    intersection = len(set(v1) & set(v2))
    denominator = np.sqrt(len(v1) * len(v2))
    return intersection / denominator


def get_similar_users(u, v, dfh):
    """
    Get the N_SIMILAR_USERS most similar users to the given one with their similarity score
    Arguments:
        u: the user_id,
        v:  the "vector" representation of the user (list of item_id)
        dfh : the "history of transaccions" dataframe

    Returns:
        tuple of lists ([similar user_id], [similarity scores])
    """
    similar_users = dfh.apply(lambda v_other: compare_vectors(v, v_other)).sort_values(ascending=False).head(
        N_SIMILAR_USERS + 1)

    if DROP_USER_FROM_HIS_NEIGHBORHOOD:
        similar_users = similar_users[similar_users.index != u]

    return similar_users.index.tolist(), similar_users.tolist()


def get_items(u, v, dfh):
    """ Get the recommend items for a given users

    It will:
        1) Get similar users for the given user
        2) Obtain all the items those users purchased
        3) Rank them using the similarity scores of the user that purchased them
        4) Return the 12 best ranked

    Arguments:
        u: the user_id,
        v:  the "vector" representation of the user (list of item_id)
        dfh : the "history of transaccions" dataframe

    Returns:
        list of item_id of lenght at most 12
    """
    global i, n

    users, scores = get_similar_users(u, v, dfh)
    df_nn = pd.DataFrame({'user': users, 'score': scores})
    df_nn['items'] = df_nn.apply(lambda row: dfh.loc[row.user], axis=1)
    df_nn['weighted_items'] = df_nn.apply(lambda row: [(item, row.score) for item in row['items']], axis=1)

    recs = pd.DataFrame(flatten(df_nn['weighted_items'].tolist()), columns=['item', 'score']).groupby('item')[
        'score'].sum().sort_values(ascending=False)
    if DROP_PURCHASED_ITEMS:
        recs = recs[~recs.index.isin(v)]
    # Keep the first 12 and get the item_ids
    i += 1
    if i % 200 == 0:
        pid = mp.current_process().pid
        print(f"[PID {pid:>2d}] Finished {i:3d} / {n:5d} - {i / n * 100:3.0f}%")
    return recs.head(12).index.tolist()


def get_items_chunk(user_ids: np.array, dfh: pd.DataFrame):
    """ Call get_item for a list of user_ids

    Arguments:
        user_ids: list of user_id,
        dfh: the "history of transaccions" dataframe

    Returns:
        pd.Series with index user_id and list of item_id (recommendations) as value
    """
    global i, n
    i = 0

    n = len(user_ids)
    pid = mp.current_process().pid
    print(f"[PID {pid:>2d}] Started working with {n:5d} users")

    df_user_vectors = pd.DataFrame(dfh.loc[user_ids]).reset_index()
    df_user_vectors['recs'] = df_user_vectors.apply(lambda row: get_items(row.user_id, row.item_id, dfh), axis=1)
    return df_user_vectors.set_index('user_id')['recs']


def get_recommendations(users: list, dfh: pd.DataFrame):
    """
    Obtained recommendation for the users using transaccion dfh in a parallelized manner

    Call get_items_chunk in a "smart" multiprocessing fashion

    Arguments:
        users: list of user_id
        dfh: the "history of transaccions" dataframe

    Returns:
        pd.DataFrame with index user_id and list of item_id (recommendations) as value

    """
    time_start = time.time()

    # Split into approximately evenly sized chunks
    # We will send just one batch to each CPU
    user_chunks = np.array_split(users, mp.cpu_count())

    f = partial(get_items_chunk, dfh=dfh)
    with Pool(mp.cpu_count()) as p:
        res = p.map(f, user_chunks)

    df_rec = pd.DataFrame(pd.concat(res))

    elapsed = (time.time() - time_start) / 60
    print(f"Finished get_recommendations({len(users)}). It took {elapsed:5.2f} mins")
    return df_rec


def uucf(df, week=START_WEEK):
    """ Entry point for the UUCF model.

    Receive the original transactions_train.csv and a start_date and gets UUCF recommendations

    The model will not cover the full list of users, but just a subset of them.

    It will provide recommendations for users with at least MINIMUM_PURCHASES after start_date.
    It might return less than 12 recs per user.

    An ad-hoc function for filling these gaps should be used downstream.
    (See fill functionality right below)


    Arguments:
        df: The raw dataframe from transactions_train.csv
        start_date: a date

    Returns:
        a submission-like pd.DataFrame with columns [customer_id, prediction]
        'prediction' is a list and not a string though

    """
    df_small = df[df['week'] > week]
    print(f"Kept data from {week} on. Total rows: {len(df_small)}")

    # H stands for "Transaction history"
    # dfh is a series of user_id => list of item_id (the list of purchases in order)
    dfh = df_small.groupby("user_id")['item_id'].apply(lambda items: list(set(items)))
    dfh = dfh[dfh.str.len() >= MINIMUM_PURCHASES]
    if TEST_RUN:
        print("WARNING: TEST_RUN is True. It will be a toy execution.")
        dfh = dfh.head(TEST_SIZE)

    users = dfh.index.tolist()
    n_users = len(users)
    print(f"Total users in the time frame with at least {MINIMUM_PURCHASES}: {n_users}")

    df_rec = get_recommendations(users, dfh)
    return df_rec

