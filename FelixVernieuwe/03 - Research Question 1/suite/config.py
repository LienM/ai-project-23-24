import os
import pathlib


import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


# =============================================================================
#                              GENERAL CONFIG
# =============================================================================

EXPERIMENT_NAME = "score_experiments"
EXPERIMENT_DESC = "Feature baseline"

# Setup Kaggle API
PROFILE_ROOT = os.environ.get("HOME", os.environ.get("USERPROFILE", ""))
KAGGLE_PATH = pathlib.Path(PROFILE_ROOT, ".kaggle/kaggle.json").as_posix()

# Output additional information
VERBOSE = True


# =============================================================================
#                               OUTPUT CONFIG
# =============================================================================

# Environment variables
DATA_PATH = "data/"
OUTPUT_PATH = "data/submissions/"

SAVE_DATAFRAME = True

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

# If larger than 0, model will do offline scoring using MAP
AMOUNT_OF_TEST_WEEKS = 0

# How many weeks to train on
TRAINING_INTERVAL = 3

# Features to ignore (exclude from training and evaluation)
FILTER_BASE_FEATURES = [

]


# UPLOAD_TO_KAGGLE = True
# INCLUDE_MISSING_CUSTOMERS = False


# See contansts.py/all_added_features for all available features
ADDED_FEATURES = [
    'weekly_rank',
    # 'all_time_rank',
    # 'has_promotion',
    # 'price_sensitivity'
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
TRAIN_INTERVAL_CANDIDATES = [
    # "all_time_bestsellers",
    "weekly_bestsellers",
    # "age_group_bestsellers",
    "previous_purchases",
    # "new_arrivals",
]

REFERENCE_WEEK_CANDIDATES = [
    # "all_time_bestsellers",
    "weekly_bestsellers",
    # "age_group_bestsellers",
    "previous_purchases",
    # "new_arrivals",
]

MISSING_CANDIDATE_METHODS = [
    "all_time_bestsellers",
    "weekly_bestsellers",
    "age_group_bestsellers",
    "new_arrivals"
]

MISSING_CANDIDATE_METHOD = MISSING_CANDIDATE_METHODS[1]


# Amount of LightGBM estimators
N_ESTIMATORS = 8

# Maximum amount of candidates to recommend
K = 12


# Amount of candidates to generate for train set
K_TRAIN = 12

# Amount of candidates to generate for test set
K_TEST = 12

# Amount of candidates to generate for missing customers
K_MISSING = 12

# Minimum accuracy of predictions
PREDICTION_ACCURACY = 0

# Maximum age of products for candidates of type "new_arrivals"
MAX_AGE = 0




# =============================================================================
# =============================================================================
# =============================================================================




DEFAULT_CONFIG = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "EXPERIMENT_DESC": EXPERIMENT_DESC,
    "TRAINING_INTERVAL": TRAINING_INTERVAL,

    "VERBOSE": VERBOSE,

    "DATA_PATH": DATA_PATH,
    "OUTPUT_PATH": OUTPUT_PATH,
    "KAGGLE_PATH": KAGGLE_PATH,
    "SAVE_DATAFRAME": SAVE_DATAFRAME,

    "AMOUNT_OF_TEST_WEEKS": AMOUNT_OF_TEST_WEEKS,
    "FILTER_BASE_FEATURES": FILTER_BASE_FEATURES,
    "ADDED_FEATURES": ADDED_FEATURES,

    "TRAIN_INTERVAL_CANDIDATES": TRAIN_INTERVAL_CANDIDATES,
    "REFERENCE_WEEK_CANDIDATES": REFERENCE_WEEK_CANDIDATES,
    "MISSING_CANDIDATE_METHOD": MISSING_CANDIDATE_METHOD,

    "N_ESTIMATORS": N_ESTIMATORS,

    "K": K,
    "K_TRAIN": K_TRAIN,
    "K_TEST": K_TEST,
    "K_MISSING": K_MISSING,

    "MAX_AGE": MAX_AGE,

    "PREDICTION_ACCURACY": PREDICTION_ACCURACY,
}