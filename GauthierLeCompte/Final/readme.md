# Project Directory Descriptions

## Directories

### `Code`
Contains all source code for the project.

#### Files
- `similar_price_outliers.ipynb`: Generates candidates based on similar price outliers.
- `similar_article_outliers.ipynb`: Generates candidates based on similar article outliers.
- `similar_all_outliers.ipynb`: Generates candidates based on both price and article outliers.
- `average_precision.py`: Required file to run the radek_warmup.ipynb.
- `radek_warmup.ipynb`: Generates parquet data files needed for running the code.
- `radek_base.ipynb`: Radek's base implementation.

### `Submissions`
Stores Kaggle submission CSV files when `KAGGLE=True` in the code.

### `Data`
Initially empty. Must contain the following files before running `radek_warmup.ipynb`:
- `articles.csv`
- `customers.csv`
- `sample_submission.csv`
- `transactions_train.csv`

## My Contributions

Located in the `Code` folder under:
- `similar_all_outliers.ipynb`
- `similar_article_outliers.ipynb`
- `similar_price_outliers.ipynb`

### My implementations by markdowns
- **Parameters to Test Week**:
  - From 'Parameters' to 'Set validation and test week'.
- **Candidates Generation**:
  - 'Similar Price Candidates' and its subsections.
  - 'Similar Article Candidates' and its subsections.
- **Combining transactions and candidates / negative examples**: 
  - And its sub-markdowns
  - Data preprocessing, cleaning and splitting for separate models, adding to Radek's code.
- **Model Training**: 
  - And its sub-markdowns
  - Creating and fitting ranker models, making predictions for standard and frequent buyer models.
- **Create Submissions**: 
  - And its sub-markdowns
  - Submission file generation for Kaggle, top 12 predictions, and combining results.
- **Evaluate the results**:
  - And its sub-markdowns
  - Evaluating outcomes for frequent buyers and combined results, evaluating novelty.
