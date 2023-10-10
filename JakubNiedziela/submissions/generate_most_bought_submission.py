import os
import tqdm
import pandas as pd 

os.chdir('../../data')

customers = pd.read_csv('customers.csv')['customer_id'].values
transactions = pd.read_csv('transactions_train.csv', usecols=['article_id'], dtype={'article_id': str})

# Get most bought articles
most_bought = transactions['article_id'].value_counts(sort=True).index.values
recomendation = ' '.join(most_bought[:12])

# Generate random submission
# submission = dict()
# for customer in tqdm.tqdm(customers):
#     submission[customer] = ' '.join(recomendation)

# submission_df = pd.DataFrame.from_dict(submission, orient='index').reset_index()
# submission_df.columns = ['customer_id', 'prediction']
submission_df = pd.DataFrame(customers, columns=['customer_id'])
submission_df['prediction'] = recomendation

print(submission_df.head())

submission_df.to_csv('submissions/most_bought_submission.csv', index=False)