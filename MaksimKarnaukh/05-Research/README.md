# Second research question

## Overview

This is a repository for my second (main) research question:

"How effective is the use of a combination of features for generating personalized candidates and ranking them compared to a pure popularity-based approach?"

Everything in this folder (`05-Research/`) is related to this second (main) research question and **should be looked at** for the total evaluation of this project.

The **methodoly** is as follows:

1) Create the 3 features.
   - Mean_price_per_c (average purchasing price of the customer), 
   - Highest_count_ign_per_c (most bought article index_group_name of the customer),
   - Mean_age_per_a (average age of everyone that bought the article).

2) Change candidate generation by doing selection based on the 3 features.

    Per customer, out of all the unique articles, select:
   1) all candidates where highest_count_ign_per_c is equal to index_group_name.
   2) From the above, the top 50 based on smallest difference between price and mean_price_per_c.
   3) From the above, the top 12 based on smallest difference between age and mean_age_per_a.

    Result is dataframe with (12*nr_of_unique_customers) rows.

3) For evaluation: How different is what it recommends from what the baseline recommends? Does the method recommend mostly items that are frequently bought, or not at all? Use Recall metric.

4) Add features to LGBM ranker and compare scores.


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
containing the three datasets (articles.parquet, customers.parquet and transactions_train.parquet) files 
in parquet format and the sample_submission.csv file.

## Structure

The `05-Research/` folder contains the following files:

* `README.md` - this file.
* `requirements.txt` - requirements file.
* `src/` - folder containing all the code files.
  * `src/candidate_generation.py` - python file containing the candidate generation code.
  * `src/helper_functions.py` - python file containing helper functions e.g. dataset reading and performance metric functions.
  
  * `src/nb_candidates_recall_s25.py` - notebook containing the code for the recall calculation for my generated candidates (**25** per customer).
  * `src/nb_candidates_recall_s50.py` - notebook containing the code for the recall calculation for my generated candidates (**50** per customer).
  * `src/nb_candidates_recall_s100.py` - notebook containing the code for the recall calculation for my generated candidates (**100** per customer).
  * `src/nb_candidates_recall_s25_5weeks.py` - notebook containing the code for the recall calculation for my generated candidates (**25** per customer), 
  for the last five weeks as validation data.
  * `src/nb_radek_candidates_recall.py` - notebook containing the code for the recall calculation for 
  [Radek's](https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack) 
  generated bestseller candidates (**12** per customer), both for only last week as validation and the last five weeks as validation.
  * `src/nb_research.py` - notebook containing the general code for the research question. 
  This follows the same structure as [Radek's](https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack) 
  file where I thus create my three features, generate the candidates and use the ranker to eventually create a 
  submission file to submit to [kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/submissions).

## Acknowledgments
Sources used:
https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack