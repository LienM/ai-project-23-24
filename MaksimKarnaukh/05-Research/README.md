# Second research question

## Overview

This is a repository for my second (main) research question:

"How effective is the use of a combination of features for generating personalized candidates and ranking them compared to a pure popularity-based approach?"

Everything in this folder (`05-Research/`) is related to this second (main) research question and **should be looked at** for the total evaluation of this project.


## Dependencies
The dependencies are listed in the requirements.txt file. 
To install the dependencies run the following command:
```bash
pip install -r requirements.txt
```
## Usage
The notebooks are self-contained and can be run as is. The data is expected to be in the input/ directory. 
The notebook is expected to be run in a Jupyter Notebook environment.

There should be a folder called `input/` in the same directory where the `05-Research/` folder is located, 
with the three dataset (articles.parquet, customers.parquet and transactions_train.parquet) files 
in parquet format and the sample_submission.csv file.

## Structure

The `05-Research/` folder contains the following files:

* `README.md` - this file.
* `requirements.txt` - requirements file.
* `src/` - folder containing all the code files.
  * `src/candidate_generation.py` - file containing the code for feature engineering.
  * `src/helper_functions.py` - file containing the code for feature engineering.
  * `src/nb_candidates_recall.py` - file containing the code for feature engineering.
  * `src/nb_candidates_recall2.py` - file containing the code for feature engineering.
  * `src/nb_candidates_recall3.py` - file containing the code for feature engineering.
  * `src/nb_candidates_recall_5weeks.py` - file containing the code for feature engineering.
  * `src/nb_radek_candidates_recall.py` - file containing the code for feature engineering.
  * `src/nb_research.py` - file containing the code for feature engineering.

## Acknowledgments
Sources used:
https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack