import torch 
import torch.nn.functional as nn
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix

def recommender_softmax(model, dataloader, restrictions, evaluate:bool=False, top_k=5):
    mps_device = torch.device("mps")
    model.eval()
    recommendations = torch.zeros(size=(0,top_k)).to(mps_device)
    model = model.to(mps_device)
    correct = 0
    total = 0

    with torch.no_grad():
        if evaluate:
            with torch.no_grad():
                for inputs, targets in tqdm(dataloader):
                    inputs = inputs.to_dense().to(mps_device)
                    targets = targets.to_dense().to(mps_device)
                    # Get predictions
                    outputs = model(inputs)
                    # Mask for articles that haven't been sold
                    mask_matrix = torch.zeros((1,outputs.shape[1])).to(mps_device)
                    mask_matrix[:,restrictions] = 1
                    results = outputs.multiply(mask_matrix)
                    # get top recommendations
                    _, top_k_indices = torch.topk(results, k=top_k, dim=1)
                    recommendations = torch.vstack([recommendations, top_k_indices])
                    # get predictions
                    predicted = torch.zeros_like(results)
                    predicted.scatter_(1, top_k_indices, 1)
                    correct_recommendations = predicted * targets
                    correct += correct_recommendations.sum().item() 
                    total += targets.sum()
            recall = correct / total
            precision = correct / (top_k*recommendations.shape[0])
            return recommendations, recall, precision
        else:
            for inputs in tqdm(dataloader):
                inputs = inputs.to_dense().to(mps_device)
                # Get predictions
                outputs = model(inputs)
                # Select top k articles
                _, top_k_indices = torch.topk(outputs, k=top_k, dim=1)
                recommendations = torch.vstack([recommendations, top_k_indices])        
            return recommendations

def recommender_two_towers(model, dataloader_cust, dataloader_art, targets, restrictions:list, evaluate: bool=False, top_k=5):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc2.out_features)).to(mps_device)
    full_customers_embeddings = torch.zeros(size=(0,model.CustomerTower.fc2.out_features)).to(mps_device)
    recommendations = torch.zeros((0,top_k))
    with torch.no_grad():
        # push customers through customer tower
        print("Generate Customer Embeddings...")
        for customers_features in tqdm(dataloader_cust):
            customers_embeddings = model.CustomerTower(customers_features.to_dense().to(mps_device))
            full_customers_embeddings = torch.vstack([full_customers_embeddings, customers_embeddings])
        # push articles through article tower
        print("Generate Articles Embeddings...")
        for articles_features in tqdm(dataloader_art):
            articles_features = model.ArticleTower(articles_features.to_dense().to(mps_device))
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    print("Get recommendations...")
    partitions = int(np.ceil(full_customers_embeddings.shape[0]/1000))
    full_articles_embeddings = full_articles_embeddings.to("cpu")
    full_customers_embeddings = full_customers_embeddings.to("cpu")
    for i in tqdm(range(partitions)):
        customer = full_customers_embeddings[i*1000:(i+1)*1000]
        predictions = nn.sigmoid(customer.matmul(full_articles_embeddings.T))
        # get rid of already bought articles
        # results = predictions - torch.tensor(targets[i*1000:(i+1)*1000].todense())
        # apply mask for products that are currently selling
        for restriction in restrictions:
            mask_matrix = torch.zeros((1,full_articles_embeddings.shape[0]))
            mask_matrix[:,restriction] = 1
            results = predictions.multiply(mask_matrix)
        _, top_k_indices = torch.topk(results, k=top_k, dim=1)
        recommendations = torch.vstack([recommendations, top_k_indices])
        recommendations = recommendations.to(torch.int64)
    if evaluate:
        predicted = torch.zeros((full_customers_embeddings.shape[0],full_articles_embeddings.shape[0]))
        predicted.scatter_(1, recommendations, 1)
        predicted = csr_matrix(predicted)
        correct_recommendations = predicted.multiply(targets)
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        recall = total_correct / total
        precision = total_correct / (top_k*predicted.shape[0])
        return recommendations, recall, precision
    else:
        return recommendations

def recommender_two_towers_embedded(model, dataloader_cust, dataloader_art, targets, restrictions, evaluate: bool=False, top_k=5):
    model = model
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc1.out_features))
    full_customers_embeddings = torch.zeros(size=(0,model.CustomerTower.fc2.out_features))
    recommendations = torch.zeros((0,top_k))
    with torch.no_grad():
        # push customers through customer tower
        print("Generate Customer Embeddings...")
        for customers_features in tqdm(dataloader_cust):
            customers_embeddings = model.CustomerTower(customers_features.to_dense())
            full_customers_embeddings = torch.vstack([full_customers_embeddings, customers_embeddings])
        # push articles through article tower
        print("Generate Articles Embeddings...")
        for articles_features in tqdm(dataloader_art):
            articles_features = model.ArticleTower(articles_features.to_dense().to(torch.int64))
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    print("Get recommendations...")
    partitions = int(np.ceil(full_customers_embeddings.shape[0]/1000))
    full_articles_embeddings = full_articles_embeddings
    full_customers_embeddings = full_customers_embeddings
    for i in tqdm(range(partitions)):
        customer = full_customers_embeddings[i*1000:(i+1)*1000]
        predictions = nn.sigmoid(customer.matmul(full_articles_embeddings.T))
        # get rid of already bought articles
        # results = predictions - torch.tensor(targets[i*1000:(i+1)*1000].todense())
        # apply mask for products that are currently selling
        mask_matrix = torch.zeros((1,full_articles_embeddings.shape[0]))
        mask_matrix[:,restrictions] = 1
        results = predictions.multiply(mask_matrix)
        _, top_k_indices = torch.topk(results, k=top_k, dim=1)
        recommendations = torch.vstack([recommendations, top_k_indices])
        recommendations = recommendations.to(torch.int64)
    if evaluate:
        predicted = torch.zeros((full_customers_embeddings.shape[0],full_articles_embeddings.shape[0]))
        predicted.scatter_(1, recommendations, 1)
        predicted = csr_matrix(predicted)
        correct_recommendations = predicted.multiply(targets)
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        recall = total_correct / total
        precision = total_correct / (top_k*predicted.shape[0])
        return recommendations, recall, precision
    else:
        return recommendations

def recommender_logistic(model, customers_n, dataloader_art, targets, restrictions, evaluate: bool=False, top_k=5):
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc4.out_features))
    recommendations = torch.zeros((0,top_k))
    with torch.no_grad():
        # push customers through customer tower
        # push articles through article tower
        print("Generate Articles Embeddings...")
        for articles_features in tqdm(dataloader_art):
            print()
            articles_features = model.ArticleTower(articles_features.to_dense())
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    print("Get recommendations...")
    full_articles_embeddings = full_articles_embeddings

    for i in tqdm(range(customers_n)):
        predictions = nn.sigmoid(model.customer_linear_layers[i](full_articles_embeddings))
        # get rid of already bought articles
        # results = predictions - torch.tensor(targets[i*1000:(i+1)*1000].todense())
        # apply mask for products that are currently selling
        mask_matrix = torch.zeros((1,full_articles_embeddings.shape[0]))
        mask_matrix[:,restrictions] = 1
        results = predictions.multiply(mask_matrix)
        _, top_k_indices = torch.topk(results, k=top_k, dim=1)
        recommendations = torch.vstack([recommendations, top_k_indices])
        recommendations = recommendations.to(torch.int64)
    if evaluate:
        predicted = torch.zeros((customers_n,full_articles_embeddings.shape[0]))
        predicted.scatter_(1, recommendations, 1)
        predicted = csr_matrix(predicted)
        correct_recommendations = predicted.multiply(targets)
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        recall = total_correct / total
        precision = total_correct / (top_k*predicted.shape[0])
        return recommendations, recall, precision
    else:
        return recommendations

def recommender_two_towers_final(model, dataloader_cust, dataloader_art, targets, restrictions:list, evaluate: bool=False, top_k=5, exclude_already_bought=False):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc4.out_features)).to(mps_device)
    full_customers_embeddings = torch.zeros(size=(0,model.CustomerTower.fc1.out_features)).to(mps_device)
    recommendations = torch.zeros((0,top_k))
    with torch.no_grad():
        # push customers through customer tower
        for customers_features in dataloader_cust:
            customers_embeddings = model.CustomerTower(customers_features.to_dense().to(mps_device))
            full_customers_embeddings = torch.vstack([full_customers_embeddings, customers_embeddings])
        # push articles through article tower
        for articles_features in dataloader_art:
            articles_features = model.ArticleTower(articles_features.to_dense().to(mps_device))
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    partitions = int(np.ceil(full_customers_embeddings.shape[0]/1000))
    full_articles_embeddings = full_articles_embeddings.to("cpu")
    full_customers_embeddings = full_customers_embeddings.to("cpu")
    for i in range(partitions):
        customer = full_customers_embeddings[i*1000:(i+1)*1000]
        predictions = nn.sigmoid(customer.matmul(full_articles_embeddings.T))
        # get rid of already bought articles
        if exclude_already_bought:
            predictions = predictions - torch.tensor(targets[i*1000:(i+1)*1000].todense())
        # apply mask for products that are currently selling
        for restriction in restrictions:
            mask_matrix = torch.zeros((1,full_articles_embeddings.shape[0]))
            mask_matrix[:,restriction] = 1
            results = predictions.multiply(mask_matrix)
        _, top_k_indices = torch.topk(results, k=top_k, dim=1)
        recommendations = torch.vstack([recommendations, top_k_indices])
        recommendations = recommendations.to(torch.int64)
    if evaluate:
        predicted = torch.zeros((full_customers_embeddings.shape[0],full_articles_embeddings.shape[0]))
        predicted.scatter_(1, recommendations, 1)
        predicted = csr_matrix(predicted)
        correct_recommendations = predicted.multiply(targets)
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        recall = total_correct / total
        precision = total_correct / (top_k*predicted.shape[0])
        return recommendations, recall, precision
    else:
        return recommendations

def recommender_two_towers_customer(model, dataloader_cust, dataloader_art, targets, restrictions:list, evaluate: bool=False, top_k=5, exclude_already_bought=False):
    mps_device = torch.device("mps")
    model = model.to(mps_device)
    # Generate customers and articles embeddings
    full_articles_embeddings = torch.zeros(size=(0,model.ArticleTower.fc4.out_features)).to(mps_device)
    full_customers_embeddings = torch.zeros(size=(0,model.CustomerTower.fc4.out_features)).to(mps_device)
    recommendations = torch.zeros((0,top_k))
    with torch.no_grad():
        # push customers through customer tower
        for customers_features in dataloader_cust:
            customers_embeddings = model.CustomerTower(customers_features.to_dense().to(mps_device))
            full_customers_embeddings = torch.vstack([full_customers_embeddings, customers_embeddings])
        # push articles through article tower
        for articles_features in dataloader_art:
            articles_features = model.ArticleTower(articles_features.to_dense().to(mps_device))
            full_articles_embeddings = torch.vstack([full_articles_embeddings, articles_features])
    # calculate probability of being purchased
    partitions = int(np.ceil(full_customers_embeddings.shape[0]/1000))
    full_articles_embeddings = full_articles_embeddings.to("cpu")
    full_customers_embeddings = full_customers_embeddings.to("cpu")
    for i in range(partitions):
        customer = full_customers_embeddings[i*1000:(i+1)*1000]
        predictions = nn.sigmoid(customer.matmul(full_articles_embeddings.T))
        # get rid of already bought articles
        if exclude_already_bought:
            results = predictions - torch.tensor(targets[i*1000:(i+1)*1000].todense())
        # apply mask for products that are currently selling
        for restriction in restrictions:
            mask_matrix = torch.zeros((1,full_articles_embeddings.shape[0]))
            mask_matrix[:,restriction] = 1
            results = predictions.multiply(mask_matrix)
        _, top_k_indices = torch.topk(results, k=top_k, dim=1)
        recommendations = torch.vstack([recommendations, top_k_indices])
        recommendations = recommendations.to(torch.int64)
    if evaluate:
        predicted = torch.zeros((full_customers_embeddings.shape[0],full_articles_embeddings.shape[0]))
        predicted.scatter_(1, recommendations, 1)
        predicted = csr_matrix(predicted)
        correct_recommendations = predicted.multiply(targets)
        total_correct = correct_recommendations.sum().item()
        total  = targets.sum().item()
        recall = total_correct / total
        precision = total_correct / (top_k*predicted.shape[0])
        return recommendations, recall, precision
    else:
        return recommendations