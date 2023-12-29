import pickle
from pathlib import Path

import pandas as pd
import torch
from Dataset import HMDataset, reverse_mapping

# Select hardware device for prediction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
path = Path('../data/HMDataset')
dataset = (HMDataset(str(path)))
data = dataset[0]
num_customer, num_article = data['customer'].num_nodes, data['article'].num_nodes

# Convert heterogeneous graph to homogeneous graph
data = data.to_homogeneous().to(device)

batch_size = 8192
mask = data.edge_index[0] < data.edge_index[1]
train_edge_label_index = data.edge_index[:, mask]

# Select the model to use, look for overfitting on the training plots
# Either load the model from the end of training or from a specific checkpoint that are done every 10 epochs
epoch = 50
if epoch is None:
    model = torch.load("../data/LightGCN/model.pt")
else:
    model = torch.load("../data/LightGCN/checkpoint_{}.pt".format(epoch))

# The dataset maps the customer and article ids to integers
# starting from 0 and going up to the number of customers and articles
# We need to map them back to the original ids for the submission
articles_path = Path('../data/HMDataset/raw/articles.parquet')
customers_path = Path('../data/HMDataset/raw/customers.parquet')
article_mapping = reverse_mapping(articles_path, index_col='article_id')
customer_mapping = reverse_mapping(customers_path, index_col='customer_id')

merge = None
# Predict for each batch of customers
# Prediction for all customers at once is too memory intensive (requires 200GB of VRAM if run on GPU)
# So depending on your hardware you might need to reduce the batch size
for start in range(0, num_customer, batch_size):
    end = start + batch_size
    if end > num_customer:
        end = num_customer
    predictions = model.recommend(data.edge_index, k=12, src_index=torch.tensor([x for x in range(start, end)], dtype=torch.int32, device=device)
                                  , dst_index=torch.tensor([x for x in range(num_article)], dtype=torch.int32, device=device))
    predictions = pd.DataFrame(predictions.cpu().numpy())

    # Convert the article ids back to the original ids
    for i in range(0, 12):
        predictions.iloc[:, i] = predictions.iloc[:, i].map(article_mapping)
    # Convert the predictions to a list of lists (the expected format for other parts of the submission code)
    predictions['prediction'] = predictions.values.tolist()

    # Convert the customer ids back to the original ids
    predictions.index.name = 'customer_id'
    predictions.reset_index(inplace=True)
    predictions['customer_id'] = predictions['customer_id'] + start
    predictions.index = predictions.index + start
    predictions['customer_id'] = predictions['customer_id'].map(customer_mapping)

    # Select only the customer_id and prediction columns
    predictions = predictions.loc[:, ['customer_id', 'prediction']]
    # Merge dataframe until we have all the predictions
    if merge is None:
        merge = predictions
    else:
        merge = pd.concat([merge, predictions])

# Save the predictions to a pickle file
# This pickle file will be loaded by the code in the notebook to create the submission file
pickle.dump(merge, open("../data/LightGCN/predictions.pkl", "wb"))
