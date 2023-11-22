import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TransactionsDataset(Dataset):
    def __init__(self, transactions, padding_article, num_articles_in_sequence):
        self.transactions_df = transactions
        self.padding_article = padding_article
        self.num_articles_in_sequence = num_articles_in_sequence

    def __len__(self):
        return len(self.transactions_df)

    def __getitem__(self, idx):
        customer_id, history = self.transactions_df.iloc[idx]
        if len(history) < 12:
            history = [self.padding_article] * (self.num_articles_in_sequence - len(history)) + history
        return torch.tensor(history[-12:], dtype=torch.int32)
    

class LSTMRecommender(nn.Module):
    def __init__(self, embedding_dim, input_dim, hidden_dim, n_articles, num_layers=2, bidirectional=True, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_articles = n_articles
        self.n_directions = 2 if bidirectional else 1
        self.num_layers = num_layers

        # Embedding articles to a lower dimension
        self.embedding = nn.Embedding(n_articles, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * num_layers, n_articles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.n_directions * self.num_layers, x.size(0), self.hidden_dim, requires_grad=True, device=device)
        c0 = torch.zeros(self.n_directions * self.num_layers, x.size(0), self.hidden_dim, requires_grad=True, device=device)
        # Embed
        embedded_sequence = self.embedding(x)
        # Forward propagate LSTM
        out, (hn, cn) = self.lstm(embedded_sequence, (h0.detach(), c0.detach()))
        # Dropout
        out = self.dropout(out)
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out
        # out = F.softmax(out, dim=1)
        # return torch.max(out, dim=1)[1]