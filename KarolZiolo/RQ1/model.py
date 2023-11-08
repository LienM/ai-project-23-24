import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class MLP1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
    
class CustomerTower(nn.Module):
    def __init__(self, input_customer_dim, output_dim=3):
        super(CustomerTower,self).__init__()
        self.fc1 = nn.Linear(input_customer_dim, 5)
        self.fc2 = nn.Linear(5, output_dim)

    def forward(self, x):
        #customers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ArticleTower(nn.Module):
    def __init__(self, input_article_dim, output_dim=3):
        super(ArticleTower,self).__init__()
        self.fc1 = nn.Linear(input_article_dim, 5)
        self.fc2 = nn.Linear(5, output_dim)

    def forward(self, x):
        #customers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TwoTower(nn.Module):
    def __init__(self, input_article_dim, input_customer_dim, output_dim=3):
        super(TwoTower,self).__init__()
        # Article tower
        self.ArticleTower = ArticleTower(input_article_dim, output_dim)
        self.CustomerTower = CustomerTower(input_customer_dim, output_dim)

    def forward(self, customer_features, article_features):
        #customers
        customer_features = self.CustomerTower(customer_features)
        # articles
        article_features = self.ArticleTower(article_features)
        # return product 
        return torch.sigmoid(torch.matmul(customer_features,article_features.T).diag())

class TwoTowerBasic(nn.Module):
    def __init__(self, input_article_dim, input_customer_dim, output_dim=3):
        super(TwoTowerBasic,self).__init__()
        # Article tower
        self.afc1 = nn.Linear(input_article_dim, 5)
        self.afc2 = nn.Linear(5, output_dim)
        # Customer tower
        self.cfc1 = nn.Linear(input_customer_dim, 5)
        self.cfc2 = nn.Linear(5, output_dim)

    def forward(self, x, y):
        #customers
        x = F.relu(self.cfc1(x))
        x = self.cfc2(x)
        # articles
        y = F.relu(self.afc1(y))
        y = self.afc2(y)
        # return product 
        return torch.matmul(x,y.T).diag()
    
    
