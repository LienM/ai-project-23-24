# AI Project 23-24

## Project description

For this class, we were tasked to tackle
the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle competition.
The goal of this competition was to predict the 12 most likely articles a customer would purchase, to be evaluated
against the products the customer actually bought in the next week.

To do this, we were provided with a plethora of [data](#Data), including customer data, article data, article images and
transaction data.

## Research questions

For this project, we had to narrow down on one or more research questions to answer. The decision was made to focus on
the following:

- What is the impact of adding a season score feature to clothing using a date offset and range to calculate season
  score?
- Can we calculate the gender of articles \& customers and use this to make predictions?
- What is the impact of seasonality and gender features on the performance of the Radek baseline LGBM ranker?

## Project structure

The project is structured as follows:

- [Candidate Generation](candidate_generation%2FREADME.md)
  > These are scripts used to generate top-12 candidates
- [Exploration](exploration%2FREADME.md)
  > These are scripts that were used to explore the data (initial classes of the course)
- [Features](features%2FREADME.md)
  > These are scripts used to add features to the dataframes.
- [Plotting](plotting%2FREADME.md)
  > These scripts were used to generate plots for the presentation and report.
- [Pruning](pruning%2FREADME.md)
  > These scripts were used for pruning the data set.
- [Radek's LGBM Ranker](radek_s_lgbmranker%2FREADME.md)
  > This is a modified version of Radek's LGBM ranker, to include my own features.
- [Selection](selection%2FREADME.md)
  > These scripts provided ways to get specific (aggregated / calculated) data from the dataframes.
- [Utils](utils%2FREADME.md)
  > These are utility scripts used throughout the project.
- [main.py](main.py)
  > This is the main script used to run the simple candidate generation.

## Usage

You will need a data folder with the following structure:

```
data
└── h-and-m-personalized-fashion-recommendations
    ├── images
    │   ├── 010
    │   │   └── (...)
    │   ├── 011
    │   │   └── (...)
    │   ├── (...)
    │   └── 095
    │       └── (...)
    ├── articles.csv
    ├── customers.csv
    ├── sample_submission.csv
    └── transactions_train.csv
```

It should be placed either in the same directory as the `main.py` script, or its parent directory.

Requirements can be installed using `pip install -r requirements.txt` from the root of the project (where `main.py` is
located).

To run the simple candidate generation, run `python main.py` from the root of the project (where `main.py` is located).

To run the Radek LGBM ranker, use Jupyter to run
the [radek_s_lgbmranker.ipynb](radek_s_lgbmranker%2Fradek_s_lgbmranker.ipynb) notebook.

## Data

We received 3 datasets in csv format, along with an example submission csv. Additionally, we received a directory with
images for articles.

- Customer data (1371980 rows)
    - Id
    - Age
    - Account info (Active, Newsletter, etc...)
    - Postal code
- Article data (105542 rows)
    - Id
    - Product code
    - Name & description
    - Colour information
    - Category information (Major category, sub category, appearance, etc...)
- Article images (105100 images)
- Transaction data (31788324 rows)
    - Date
    - Customer id
    - Article id
    - Price
    - Sales channel id