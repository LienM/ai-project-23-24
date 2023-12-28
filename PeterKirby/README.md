# Ranker Ensembling AI Project

This folder contains my project submission for the AI project.

The project is structured as follows:

### Not Relevant for Research Questions
File|Description
---|---
**EDA.ipynb** | Exploratory data analysis. Not strictly used for answering the research questions, but contains some interesting insights into the data.
**FeatureEngineering.ipynb** | Brief feature engineering. Similarly to EDA, contains interesting information but not used for answering research questions.

### Helpers/Ensemble classes

Rankers folder:
File|Description
---|---
**Ranker\.py** | Ranker wrapper class used for converting scikit-learn classifiers into pointwise rankers.
**Bagger\.py** | Class written as a bagging ensembler for rankers. Written because it was necessary to have an ensemble method that can take rankers as base learners and combine results using methods developed for this project.
**Stacker\.py** | Class written as a Stacker ensembler. Takes rankers as base learners and combines them with methods developed for this project.

Other helpers:
File|Description
---|---
**Evaluations\.py** | File containing the functions used for evaluating ranking methods. Functions for MAPK, precision, recall, MRR.
**PrepareData\.py** | File containing function for generating candidates and splitting data into test and training sets in the same way as Radek's code.

### Experiments From Report
File|Description
---|---
**RankAggregation.ipynb** | Tackles the first research question "Can aggregation of multiple rankers’ rankings be used to improve the score of individual rankers?"
**Bagging.ipynb** | Attempts to answer the second research question "Can a bagger be effectively adapted to use rankers as base learners?"
**Stacking.ipynb** | Investigates the third research question "Can the SuperLearner be adapted to use rankers as base learners, and produce a higher scoring ranking than that of its constituent rankers?"
**StackingBagging.ipynb** | Start of experiment looking into the idea of splitting dataset by values of a given category before giving to base rankers to train a stacker. Not thorough due to time constraints.
**KaggleSubmissions.ipynb** | Notebook used for generating Kaggle submissions. Not well commented as it contains the same logic from other notebooks, but trained/tested on shifted data, and results submitted to Kaggle.


### Experiments Not in Report
File|Description
---|---
**Stacking_VaryingValidation.ipynb** | Brief experiment with varying the number of training weeks to test rank averaging stacker models. Experiment is not thorough due to time constraints, but indicates that different models benefit from different numbers of training/validation weeks.
