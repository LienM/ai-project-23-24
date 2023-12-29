import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class MLP1(nn.Module):
    '''MLP model with 1 hidden layer'''
    def __init__(self, input_dim, output_dim):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class MLP2(nn.Module):
    '''MLP model with 2 hidden layers'''
    def __init__(self, input_dim, output_dim):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
    
class CustomerTower(nn.Module):
    '''Customer Tower model with 1 hidden layer'''
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
    '''Article Tower model with 1 hidden layer'''
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
    '''Two Tower model with shallow Customer Tower and Article Tower'''
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

class ArticleTowerEmbedded(nn.Module):
    '''Article Tower with embedded layers'''
    def __init__(self, article_cat_dim, embedding_dim=3, output_dim=3):
        super(ArticleTowerEmbedded, self).__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories in article_cat_dim
        ])
        self.fc1 = nn.Linear(embedding_dim * len(article_cat_dim), output_dim)

    def forward(self, x):
            # Embedding layers for categorical variables
            embedded_features = [
                embedding_layer(x[:,i].T)
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]

            # Concatenate embedded categorical variables along the last dimension
            x = torch.cat(embedded_features, dim=-1)

            # Fully connected layer
            x = F.relu(self.fc1(x))
            return x

class TwoTowerEmbedded(nn.Module):
    '''Two Tower model with embedded Article Tower and shallow Customer Tower'''
    def __init__(self, article_cat_dim, input_customer_dim, embedding_dim=5, output_dim=3):
        super(TwoTowerEmbedded, self).__init__()
        # Article tower
        self.ArticleTower = ArticleTowerEmbedded(article_cat_dim, embedding_dim, output_dim)
        # Customer tower
        self.CustomerTower = CustomerTower(input_customer_dim, output_dim)

    def forward(self, customer_features, article_features):
        # customers
        c_features = self.CustomerTower(customer_features)
        # articles
        a_features = self.ArticleTower(article_features)
        # return product
        return torch.sigmoid(torch.matmul(c_features, a_features.to(c_features.dtype).T).diag())

class ArticleTowerLog(nn.Module):
    '''Article Tower with deep layers'''
    def __init__(self, input_article_dim, output_dim=3):
        super(ArticleTowerLog,self).__init__()
        self.fc1 = nn.Linear(input_article_dim, 250)
        self.fc2 = nn.Linear(250, 125)
        self.fc3 = nn.Linear(125, 50)
        self.fc4 = nn.Linear(50, output_dim)

    def forward(self, x):
        #customers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class LogisticRegression(nn.Module):
    '''Logistic Regression model with deep Article Tower and seperate list of linear layers for each customer'''
    def __init__(self, input_article_dim, input_customer_dim, output_dim=3):
        super(LogisticRegression, self).__init__()

        # Article tower
        self.ArticleTower = ArticleTowerLog(input_article_dim, output_dim)

        # Linear layers for each customer
        self.customer_linear_layers = nn.ModuleList([
            nn.Linear(output_dim, 1)  # Assuming input_article_dim is the same for each customer
            for _ in range(input_customer_dim)
        ])

    def forward(self, customers_id, article_features):
        # Articles
        article_features = self.ArticleTower(article_features)
        # Individual linear layers for each customer
        selected_layers = [self.customer_linear_layers[i] for i in customers_id.to(torch.int64)]

        # Apply all linear layers to article features in a single forward pass
        customer_logits = torch.cat([layer(article_features) for layer in selected_layers], dim=1)

        # Apply sigmoid activation to get probabilities
        customer_probabilities = torch.sigmoid(customer_logits)

        return customer_probabilities

class CustomerTowerFinal(nn.Module):
    '''Customer Tower model with 1 layer'''
    def __init__(self, input_customer_dim, output_dim=10):
        super(CustomerTowerFinal,self).__init__()
        self.fc1 = nn.Linear(input_customer_dim, output_dim)

    def forward(self, x):
        #customers
        x = self.fc1(x)
        return x
    
class ArticleTowerFinal(nn.Module):
    '''Article Tower model with 3 hidden layers'''
    def __init__(self, input_article_dim, output_dim=10):
        super(ArticleTowerFinal,self).__init__()
        self.fc1 = nn.Linear(input_article_dim, 250)
        self.fc2 = nn.Linear(250, 125)
        self.fc3 = nn.Linear(125, 50)
        self.fc4 = nn.Linear(50, output_dim)

    def forward(self, x):
        #customers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class TwoTowerFinal(nn.Module):
    '''Two Tower model with shallow Customer Tower and deep Article Tower'''
    def __init__(self, input_article_dim, input_customer_dim, output_dim=10):
        super(TwoTowerFinal,self).__init__()
        # Article tower
        self.ArticleTower = ArticleTowerFinal(input_article_dim, output_dim)
        self.CustomerTower = CustomerTowerFinal(input_customer_dim, output_dim)

    def forward(self, customer_features, article_features):
        #customers
        customer_features = self.CustomerTower(customer_features)
        # articles
        article_features = self.ArticleTower(article_features)
        # return product 
        return torch.sigmoid(torch.matmul(customer_features,article_features.T).diag())

class CustomerTowerDiversification(nn.Module):
    '''Customer Tower model with 3 hidden layers'''
    def __init__(self, input_customer_dim, output_dim=10):
        super(CustomerTowerDiversification,self).__init__()
        self.fc1 = nn.Linear(input_customer_dim, 250)
        self.fc2 = nn.Linear(250, 125)
        self.fc3 = nn.Linear(125, 50)
        self.fc4 = nn.Linear(50, output_dim)

    def forward(self, x):
        #customers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class TwoTowerCustomer(nn.Module):
    '''Two Tower model with deep Customer Tower and Article Tower'''
    def __init__(self, input_article_dim, input_customer_dim, output_dim=10):
        super(TwoTowerCustomer,self).__init__()
        # Article tower
        self.ArticleTower = ArticleTowerFinal(input_article_dim, output_dim)
        self.CustomerTower = CustomerTowerDiversification(input_customer_dim, output_dim)

    def forward(self, customer_features, article_features):
        #customers
        customer_features = self.CustomerTower(customer_features)
        # articles
        article_features = self.ArticleTower(article_features)
        # return product 
        return torch.sigmoid(torch.matmul(customer_features,article_features.T).diag())
