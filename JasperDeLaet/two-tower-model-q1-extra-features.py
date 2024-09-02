# imports
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random

# For encoding non-numeric features
class LabelEncoder:
    def __init__(self, labels):
        unique_labels = set(labels)
        self.vocab_size = len(unique_labels)

        self.label_to_index = {"unknown": 0}
        for idx, label in enumerate(unique_labels):
            self.label_to_index[label] = idx
        self.unknown_index = self.label_to_index["unknown"]

        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def encode(self, labels):
        return torch.tensor([self.label_to_index.get(label, self.unknown_index) for label in labels], dtype=torch.long)

    def decode(self, indices):
        return [self.index_to_label.get(idx, self.index_to_label[self.unknown_index]) for idx in indices]


# Hyperparameters (to fine-tune)

# learning rate
lr = 0.02

# batch size
batch_size = 512

# embedded dimension
embed_dim = 128

# load data in dataframes

print('Load dataset ...')

article_df = pd.read_csv("./project_antwerp/AIProject/data/articles.csv")
customer_df = pd.read_csv("./project_antwerp/AIProject/data/customers.csv")
transactions_df = pd.read_csv("./project_antwerp/AIProject/data/transactions_train.csv")

customer_features = ['age']

# split data
print('Splitting datasets ...')

train_df = transactions_df[(transactions_df['t_dat'] >= '2020-06-01') & (transactions_df['t_dat'] <= '2020-09-14')]
test_df = transactions_df[(transactions_df['t_dat'] > '2020-09-15')]

print('Making encoders ...')

customer_encoder = LabelEncoder(train_df['customer_id'].unique())
article_encoder = LabelEncoder(train_df['article_id'].unique())

customer_ids_in_train = train_df['customer_id'].unique()
article_ids_in_train = train_df['article_id'].unique()

customer_df = customer_df[customer_df['customer_id'].isin(customer_ids_in_train)]
article_df = article_df[article_df['article_id'].isin(article_ids_in_train)]


# Custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, transactions, all_article_ids):
        self.transactions = transactions
        self.all_article_ids = list(all_article_ids)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.transactions)

    def __getitem__(self, index: int):
        row = self.transactions.iloc[index]
        customer_id = row['customer_id']
        positive_article_id = row['article_id']
        # Negative sampling
        negative_article_id = random.choice(self.all_article_ids)
        while negative_article_id == positive_article_id:
            negative_article_id = random.choice(self.all_article_ids)

        customer_row = customer_df[customer_df['customer_id'] == customer_id]
        age = customer_row['age'].values[0]

        return customer_id, age, positive_article_id, negative_article_id


train_data = CustomDataset(train_df[['customer_id', 'article_id']], train_df['article_id'].unique())
test_data = CustomDataset(test_df[['customer_id', 'article_id']], train_df['article_id'].unique())

# dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

print('Train length: ' + str(len(train_loader)))
print('Test length: ' + str(len(test_loader)))


# Create Two tower model using PyTorch
class TwoTower(nn.Module):
    # In our case, items are articles and users are customers
    def __init__(self, n_users, n_items):
        super(TwoTower, self).__init__()

        self.user_embedding = nn.Embedding(num_embeddings=n_users, embedding_dim=embed_dim)
        self.item_embedding = nn.Embedding(num_embeddings=n_items, embedding_dim=embed_dim)
        self.age_linear = nn.Linear(1, embed_dim)

        # 2 linear layers
        self.user_layers = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, embed_dim, bias=True),
            nn.LeakyReLU()
        )

        # 2 linear layers
        self.item_layers = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, embed_dim, bias=True),
            nn.LeakyReLU()
        )

        self.dot_product = torch.matmul

    def forward(self, users, ages, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        age_embedding = self.age_linear(ages.unsqueeze(1).float())
        user_embedding = user_embedding + age_embedding

        user_embedding = self.user_layers(user_embedding)
        item_embedding = self.item_layers(item_embedding)

        return self.dot_product(user_embedding, item_embedding.t())


# Device selection
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Model
model = TwoTower(customer_encoder.vocab_size, article_encoder.vocab_size)
model = model.to(device)


def train(model, device, dataloader, optimizer):
    model.train()
    train_loss = 0
    for batchidx, data in enumerate(dataloader):
        if batchidx % 100 == 0:
            print('TRAIN Batch: ' + str(batchidx) + '/' + str(len(dataloader)))
        customers, ages, positive_articles, negative_articles = data
        customers = customer_encoder.encode(customers)
        positive_articles = article_encoder.encode(positive_articles.tolist())
        negative_articles = article_encoder.encode(negative_articles.tolist())
        ages = torch.tensor(ages, dtype=torch.float32)

        optimizer.zero_grad()

        customers = customers.to(device)
        ages = ages.to(device)
        positive_articles = positive_articles.to(device)
        negative_articles = negative_articles.to(device)

        positive_predictions = model(customers, ages, positive_articles)
        negative_predictions = model(customers, ages, negative_articles)

        # Form of Hinge loss
        loss = torch.mean(torch.relu(1.0 - positive_predictions + negative_predictions))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss


def evaluation(model, device, dataloader):
    model.eval()
    top_1000_correct = 0
    top_500_correct = 0
    top_100_correct = 0
    with torch.no_grad():
        for batchidx, data in enumerate(dataloader):
            if batchidx % 100 == 0:
                print('EVAL Batch: ' + str(batchidx) + '/' + str(len(dataloader)))
            customers, positive_articles, negative_articles = data
            customers = customer_encoder.encode(customers)
            positive_articles = article_encoder.encode(positive_articles.tolist())
            all_articles = article_encoder.encode(article_encoder.label_to_index.keys())
            ages = torch.tensor(ages, dtype=torch.float32)

            # all articles -> to get accurate recall values
            customers = customers.to(device)
            ages = ages.to(device)
            positive_articles = positive_articles.to(device)
            all_articles = all_articles.to(device)

            outputs = model(customers, ages, all_articles)

            positive_predictions = model(customers, ages, positive_articles)

            # Top k=1000
            top_indices = torch.topk(outputs, 1000).indices
            for i, article in enumerate(positive_articles):
                if article in top_indices[i].tolist():
                    top_1000_correct += 1
                if article in top_indices[i][:500].tolist():
                    top_500_correct += 1
                if article in top_indices[i][:100].tolist():
                    top_100_correct += 1
    return (top_1000_correct / len(dataloader.dataset), top_500_correct / len(dataloader.dataset),
            top_100_correct / len(dataloader.dataset))


# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)


train_losses = []
val_losses = []
top_100_recall_lst = []
top_500_recall_lst = []
top_1000_recall_lst = []

print('Start training ...')

# Train and evaluation loop
epochs = 4
for epoch in range(epochs):
    print("Epoch: " + str(epoch) + "/" + str(epochs))
    train_loss = train(model, device, train_loader, optimizer)
    top_1000_recall, top_500_recall, top_100_recall = evaluation(model, device, test_loader)
    train_losses.append(train_loss)
    top_100_recall_lst.append(top_100_recall)
    top_500_recall_lst.append(top_500_recall)
    top_1000_recall_lst.append(top_1000_recall)
    print("Train loss: " + str(train_loss))
    print("Top 100 recall: " + str(top_100_recall))
    print("Top 500 recall: " + str(top_500_recall))
    print("Top 1000 recall: " + str(top_1000_recall))


print(train_losses)
print(val_losses)
print(top_100_recall_lst)
print(top_500_recall_lst)
print(top_1000_recall_lst)

print('end')


