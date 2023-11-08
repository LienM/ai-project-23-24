import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Define the training function for multi-label classification with validation
def train_softmax(model, train_dataloader, val_dataloader, criterion, optimizer, save_dir, num_epochs=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    val_loss_list = []
    min_loss = np.inf
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs, targets in train_dataloader:
            inputs = inputs.to_dense().to(mps_device)
            targets = targets.to_dense().to(mps_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()       
        # Validatete for the epoch
        train_loss = loss.item()
        val_loss = validate_softmax(model, val_dataloader, criterion)
        val_loss_list.append(val_loss) 
        if val_loss<min_loss:
            min_loss = val_loss
            torch.save(model, save_dir)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}%')
    return val_loss_list

# Define the validation function for multi-label classification
def validate_softmax(model, val_dataloader, criterion, k=5):
    mps_device = torch.device("mps")
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to_dense().to(mps_device)
            targets = targets.to_dense().to(mps_device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    return val_loss

# Define the training function for multi-label classification with validation
def train_two_tower(model, customers, articles, train_dataloader, val_dataloader, criterion, optimizer, save_dir, num_epochs=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    val_loss_list = []
    min_loss = np.inf
    for epoch in range(num_epochs):
        model.train()
        for articles_id, customers_id, targets in tqdm(train_dataloader): 
            # Positive sample
            articles_features = torch.tensor(articles[articles_id].todense(), dtype=torch.float32)
            customer_features = torch.tensor(customers[customers_id].todense(), dtype=torch.float32)
            articles_features = articles_features.to(mps_device)
            customer_features = customer_features.to(mps_device)
            targets = targets.to(mps_device)
            # Pus to model
            optimizer.zero_grad()
            outputs = model(customer_features, articles_features)
            # Generate outputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        # Validatete for the epoch
        train_loss = loss.item()
        val_loss = validate_two_tower(model, val_dataloader, articles, customers, criterion)
        val_loss_list.append(val_loss)
        if val_loss<min_loss:
            min_loss = val_loss
            torch.save(model, save_dir)
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    return val_loss_list

# Define the validation function for multi-label classification
def validate_two_tower(model, val_dataloader, articles, customers, criterion):
    mps_device = torch.device("mps")
    model.eval()
    val_loss = 0.0
    model = model.to(mps_device)
    with torch.no_grad():
        for articles_id, customers_id, targets in val_dataloader:
            articles_features = torch.tensor(articles[articles_id].todense(), dtype=torch.float32).to(mps_device)
            customer_features = torch.tensor(customers[customers_id].todense(), dtype=torch.float32).to(mps_device)
            targets = targets.to(mps_device)
            outputs = model(customer_features, articles_features)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    val_loss /= len(val_dataloader)
    return val_loss






