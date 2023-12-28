import os
import tqdm
import pandas as pd 
import numpy as np

os.chdir('../../data')

customers = pd.read_csv('customers.csv')['customer_id'].values
transactions = pd.read_csv('transactions_train.csv', usecols=['article_id'], dtype={'article_id': str})

# Get most bought articles
most_bought = transactions['article_id'].value_counts(normalize=True, sort=True)
probabilities = most_bought.values
most_bought = most_bought.index.values

# Generate most bought submission
submission_df = pd.DataFrame(customers, columns=['customer_id'])
submission_df['prediction'] = ''
for i, row in submission_df.iterrows():
    if i % 10000 == 0:
        print(i)
    row['prediction'] = ' '.join(np.random.choice(most_bought, 12, p=probabilities))

print(submission_df.head())

submission_df.to_csv('submissions/most_bought_submission_with_prob.csv', index=False)