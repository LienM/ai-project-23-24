# Background information

- Author: Sam Roggeman
- Course: Artificial Intelligence Project
- Year: 2023-2024
- Institution: Universiteit Antwerpen

# Info

This repository contains the code for the project of the course Artificial Intelligence Project. The goal of this
project is to create a ranker based on a multilayer perceptron for ranking items. 

# Dataset
For this project, the H&M Personalized Fashion Recommendations dataset is used. This dataset can be found on kaggle
via the following link: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data. 

# File structure
The file structure of this repository is as follows:
```
├── Input 
│ └── Dataset
├── models
├── output
└── Project (repository root)
  └── SamRoggeman
    ├── requirements.txt
    └── src

```

# Installation (Only once)
- Download the repository
- Install the requirements from the requirements.txt file
- Download the data from kaggle
- Unzip the data and place it in the data folder which is located at in the parent directory of the repository as seen in the filestructure
- Run the solution_warmup.ipynb file from src.RADEK 
- Run the custom data_warmup.ipynb file from src

# Usage
Now that the data is preprocessed, the model can be trained. This can be done by running the train_models.py file.
This file will train and evaluate the models and save them in the models folder. It will return the evaluation metrics (map@k) for each model.

One could change the set of models that are trained by changing min, max and stepsize of both the number of layers and the number of neurons per layer inside the train_models function in train_models.py file.

Now that the models are trained, they can be used to predict the ranking of the test data. This can be done by running
generate_predictions.py. This file will generate the predictions and save them in the output folder.

Hyperparameter tuning can be done by running the hyperparameter_tuning.py file. 
This file will tune the hyperparameters as specified in the file and save the results in the output folder.
Although this file is not used in the final solution, it is still included in the repository for completeness as it can be used to tune the hyperparameters after adapting the size of the multilayer perceptron (it is 1 hidden layer of 64 neurons).


# Example
An example of the full usage of the code can be found in the NN_Ranking.ipynb file. 
This file will run the entire pipeline from training to generating the predictions.

# Results
The resulting prediction file of the model can be found in the output folder.
There is also a image generated that shows the map@k for each model. This image can be found in the output folder as well.
