import os
import pathlib

import logging

#  GENERAL NOTE: (un)comment a line to enable/disable it


# =============================================================================
#                              GENERAL CONFIG
# =============================================================================

# Name and description that should be used for the Kaggle submission
EXPERIMENT_NAME = "score_experiments"
EXPERIMENT_DESC = "Feature baseline"

# Setup Kaggle API
PROFILE_ROOT = os.environ.get("HOME", os.environ.get("USERPROFILE", ""))
KAGGLE_PATH = pathlib.Path(PROFILE_ROOT, ".kaggle/kaggle.json").as_posix()

# Output additional information
VERBOSE = False

# =============================================================================
#                               OUTPUT CONFIG
# =============================================================================

# Output paths
DATA_PATH = "data/"
OUTPUT_PATH = "data/submissions/"

# Whether to generally save the submission to a csv
SAVE_DATAFRAME = False

# As long as the above folders can't be found, move up one directory
ctr = 0
while not os.path.exists(OUTPUT_PATH):
    DATA_PATH = "../" + DATA_PATH
    OUTPUT_PATH = "../" + OUTPUT_PATH
    ctr += 1
    if ctr > 5:
        raise Exception("Could not find data and output paths.")

# =============================================================================
#                              EXPERIMENT CONFIG
# =============================================================================

#                    TRAIN WEEKS            TEST WEEKS           TEST OFFSET
# __________//***********************||+++++++++++++++++++++//_______________||
# ^ DATA START                 REFERENCE WEEK                       DATA END ^


# For offline evaluation, how many weeks of ground truth candidates can be compared against
TEST_WEEKS = 0

# For offline evaluation, how many weeks of the data should be excluded from the end
#   e.g. For TEST_OFFSET = 24, you can test the behaviour of the candidate generation in a different season
TEST_OFFSET = 0

# For candidate generation, how many weeks of the user's purchase history can be used
# TODO: Can be extended into different methods
#    1) Fetch more weeks of the user's purchase history if they are very active customers (reduce computation time)
#    2) Determine an optimal cut-off point for the training data
TRAIN_WEEKS = 3

# Features to ignore (exclude from training and evaluation)
FILTER_BASE_FEATURES = [

]

# See contansts.py/all_added_features for all available features
# Custom features to add
ADDED_FEATURES = [
    'weekly_rank',
    # 'all_time_rank',
    # 'price_sensitivity',
    # 'new_arrival',
    # 'has_promotion',
    # 'age_group',
]

# See constants.py/all_candidate_methods for all available candidate generation methods
REGULAR_CANDIDATE_METHODS = [
    "all_time_bestsellers",
    "weekly_bestsellers",
    "age_group_bestsellers",
    "new_arrivals"
    "previous_purchases",
]

# Candidates to add
TRAIN_CANDIDATE_METHODS = [
    # {"type": "all_time_bestsellers", "k": 20},
    {"type": "weekly_bestsellers", "k": 12},
    # {"type": "age_group_bestsellers", "k": 12},
    # {"type": "new_arrivals", "max_age": 0, "k": 12},
    # {"type": "previous_purchases", "k": 12},
]

TEST_CANDIDATE_METHODS = [
    # {"type": "all_time_bestsellers", "k": 20},
    {"type": "weekly_bestsellers", "k": 12},
    # {"type": "age_group_bestsellers", "k": 12},
    # {"type": "new_arrivals", "max_age": 0, "k": 12},
    # {"type": "previous_purchases", "k": 12},
]

MISSING_CANDIDATE_METHODS = [
    {"type": "all_time_bestsellers", "k": 12},
    {"type": "weekly_bestsellers", "k": 12},
    {"type": "age_group_bestsellers", "k": 12},
    {"type": "new_arrivals", "max_age": 0, "k": 12},
]

MISSING_CANDIDATE_METHOD = MISSING_CANDIDATE_METHODS[1]

# Amount of LightGBM estimators
N_ESTIMATORS = 8

# Maximum amount of candidates to recommend
K = 12

# Minimum accuracy of predictions
PREDICTION_ACCURACY = 0.0

# =============================================================================
# =============================================================================
# =============================================================================


DEFAULT_CONFIG = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "EXPERIMENT_DESC": EXPERIMENT_DESC,

    "VERBOSE": VERBOSE,

    "DATA_PATH": DATA_PATH,
    "OUTPUT_PATH": OUTPUT_PATH,
    "KAGGLE_PATH": KAGGLE_PATH,
    "SAVE_DATAFRAME": SAVE_DATAFRAME,
    "SAVE_SCORE": True,

    "TRAIN_WEEKS": TRAIN_WEEKS,
    "TEST_WEEKS": TEST_WEEKS,
    "TEST_OFFSET": TEST_OFFSET if TEST_WEEKS > 0 else 0,

    "FILTER_BASE_FEATURES": FILTER_BASE_FEATURES,
    "ADDED_FEATURES": ADDED_FEATURES,

    "TRAIN_CANDIDATE_METHODS": TRAIN_CANDIDATE_METHODS,
    "TEST_CANDIDATE_METHODS": TEST_CANDIDATE_METHODS,
    "MISSING_CANDIDATE_METHOD": MISSING_CANDIDATE_METHOD,

    "N_ESTIMATORS": N_ESTIMATORS,

    "K": K,

    "PREDICTION_ACCURACY": PREDICTION_ACCURACY,
}


logging.basicConfig(
    level=logging.DEBUG if VERBOSE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
print = logging.debug
