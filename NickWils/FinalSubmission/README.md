# How to run the code

This folder contains finalized notebooks organized by research question. The code was developed and run on Kaggle; however, to run it locally, follow the steps below:

1. Download the files from the Kaggle competition.
2. Ensure proper execution order, with the preprocessing notebook "Base_Code/make_parquet.ipynb" being run first, as this will create the used input files.
3.  Change input files to your file location as indicated in each notebook. Most input files will be located at the start of the notebook or at the end when a submission is created.


# Project Structure

## Base Code
This folder contains general code used in all research questions.

- **make_parquet.ipynb**: Preprocessing notebook creating parquet files used throughout most notebooks.
- **making_graphs.ipynb**: Creates graphs with manually filled-in Kaggle scores.
- **ranker.ipynb**: Altered Radek notebook implementation.
- **test-metrics.ipynb**: Notebook used to test the used metrics.

## Research Question Notebooks

### RQ_Similarity

- **similar_article_analysis.ipynb**: Analyzes the dataset for item similarity.
- **similarity_ranker.ipynb**: Uses the analysis to create predictions.

### RQ_Seasonality

- **season_analysis.ipynb**: Analyzes the dataset for seasonality.
- **season_boosted_ranker.ipynb**: Uses the analysis to create predictions.

### RQ_Repurchase_Data

#### Analysis

- **repurchase_analysis.ipynb**: Analysis of the repurchase data.
- **repurchase_relevance_predictions.ipynb**: Analysis of the relevance of the repurchase data on Radek's notebook.

#### Baseline

- **Popularity**: Folder containing notebooks experimenting on Popularity.
- **OwnedItems**: Folder containing notebooks experimenting on repurchase data.
- **Combinations**: Folder containing notebooks experimenting on Popularity and repurchase data.

#### Candidate Generation

- **candidate-repurchase.ipynb**: Radek's notebook with my repurchase candidate generation.
- **candidate-repurchase-meanprice.ipynb**: Radek's notebook with my repurchase and mean price candidate generation.
- **candidate-radek-repurchase-meanprice.ipynb**: Radek's notebook with my repurchase with mean price and merged with Radek's candidate generation.