# Project AI code

This directory contains the code used in my project for the project AI course (2023) at the university of Antwerp.

## Overview

`assignment1_EDA.ipynb` and `assignment2_FE.ipynb` contain code related to the first two assignments of the course. These were not used for answering the research questions beyond the insights into the data they uncovered.

`radek_preprocessing.ipynb` contains code to preprocess the raw H&M dataset and make it more easy to work with. The results of this preprocessing are saved as parquet files and referenced in most other notebooks.

The `template/` directory contains a generic template for performing vaidation experiments and generating submission files using a simple candidate generation scheme and an LGBM ranker. It can be viewed as an upgraded version of [Radek's LGBMRanker starter-pack](https://www.kaggle.com/code/marcogorelli/radek-s-lgbmranker-starter-pack). The notebook and accompanying python script was shared on the course discord for other people to use

`candidates.ipynb` and `main.ipynb` contain the main code used for the experiments in the report. The first notebook looks at different candidate generation methods and measures their recall. The second notebook applies the candidate generation methods to the experiment template using a ranker as discussed above. Both notebooks use the `candidate_generation.py` script which contains all the code for the actual candidate generation.

`LLM.ipynb` contains code to fine tune GPT-2 based on the GPT4Rec approach to candidate generation. Its output is used in one of the candidate generation methods.

`recpack_experiments.ipynb` contains code which performs experiments to see which collaborative filtering algroithms would be best to use on the H&M dataset. It is powered by [recpack](https://recpack.froomle.ai/). The second half of the notebook also exports a dataframe containing the top-100 most similar item to each other item. This dataframe is used in one of the candidate generation methods.

`probing.ipynb` contains the code and result of probing experiments I performed for cold customers.

## Tests

The template script and some of the candidate generation functions are tested using pytest in `template/experiment_template_test.py` and `candidate_generation_test.py` respectively.
