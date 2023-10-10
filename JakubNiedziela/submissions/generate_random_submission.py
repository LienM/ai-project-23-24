import os
import tqdm
import pandas as pd 
from random import randint

os.chdir('../../data')

customers = pd.read_csv('customers.csv')['customer_id'].values
articles = pd.read_csv('articles.csv', dtype={'article_id':str})['article_id']


# Generate random submission
submission = dict()
for customer in tqdm.tqdm(customers):
    n_to_predict = 12
    articles_to_predict = articles.sample(n_to_predict).values
    submission[customer] = ' '.join(articles_to_predict)

submission_df = pd.DataFrame.from_dict(submission, orient='index').reset_index()
submission_df.columns = ['customer_id', 'prediction']

print(submission_df.head())

submission_df.to_csv('submissions/random_submission.csv', index=False)