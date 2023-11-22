import numpy as np
from scipy.stats import rankdata
from Evaluations import get_preds_and_actual, mrr, mapk, precision_recall_at_k
import pandas as pd

class Stacker():
    '''
    Class for ensembling rankers - sklearn has stacking class for classifiers and regressors but cannot pass rankers as base estimators.
    
    Each base estimator is trained on the whole dataset, then final ranker is trained on cross-validated predictions (as with sklearn stackers).
    '''

    def __init__(self, rankers, meta_learner, use_groups=None):
        '''
        Parameters:
            rankers: a list of instances of base rankers that are used to create stacked ranker.
            meta_learner: an instance of the model used to combine results of the weak rankers.
            use_groups (list or None): a list of booleans indicating which rankers should have data split into groups. If none, none of the rankers will.
        '''

        self.rankers = rankers
        self.ranker_weights = {
            "MRR": [],
            "MAPk": [],
            "Pk": []
        }
        #self.ranker_MRRs = []
        #self.ranker_MAPks = []
        #self.ranker_Pks = []


        self.meta_learner = meta_learner

        self.use_groups = use_groups
        


    def fit(self, train, columns_to_use):
        '''
        Computes ranker weights by splitting training set into training/validation and judging performance on validation set, then fits all of the base rankers to the whole dataset.
               
        Parameters
            train (pd.dataframe): The dataframe of training data - each row is a sample, each column represents a feature of the data.
            columns_to_use (list): list of strings of column headers which are to be used to train the rankers.
            bestsellers_previous_week (None or pandas.DataFrame): frame of the bestsellers from the previous week
        
        '''

        #1) train all classifiers on all but last week of TRAINING data.
        #2) using these classifiers, predict purchases for last week of TRAINING data - record MRR, MAP@k, precision@k.
        #3) retrain rankers on full training set

        #computing ranker weights by splitting training data into train and validation set - get weights from performance on predictions of validation set
        validation_week = train.week.max()
        validation = train[train.week == validation_week].copy()
        val_x = validation[columns_to_use]
        validation_transactions = validation[validation['purchased'] == 1]

        print("Computing weights...")

        #split for training when doing validation
        val_train = train[train.week != validation_week].copy()

        val_train_x = val_train[columns_to_use]
        val_train_y = val_train['purchased']

        print("    Fitting to validation training set...")

        if self.use_groups is None:
            for ranker in self.rankers:
                ranker.fit(val_train_x, val_train_y)
            
        else:
            val_train_baskets = val_train.groupby(['week', 'customer_id'])['article_id'].count().values
            for i in range(len(self.rankers)):
                if self.use_groups[i]:
                    self.rankers[i].fit(val_train_x, val_train_y, group=val_train_baskets)

                else:
                    self.rankers[i].fit(val_train_x, val_train_y)



        print("    Testing on validation set...")
        #computing "weights" for each ranker
        self.ranker_weights = {
            "MRR": [],
            "MAPk": [],
            "Pk": []
        }

        for i in range(len(self.rankers)):
            validation[f"ranker_{i}"] = self.rankers[i].predict(val_x)
            #look at what was actually bought in the validation week, make list of actual purchases, and get scores
            preds, actual_purchases, preds_dense, actual_purchases_dense = get_preds_and_actual(validation, f"ranker_{i}", validation_transactions, pd.read_csv('../../../Data/sample_submission.csv'))

            self.ranker_weights["MRR"].append(mrr(actual_purchases_dense, preds_dense))
            self.ranker_weights["MAPk"].append(mapk(actual_purchases_dense, preds_dense, k=12))
            self.ranker_weights["Pk"].append(precision_recall_at_k(actual_purchases_dense, preds_dense, k=12)[0])




        #retraining rankers on the whole training data
        print("Retraining rankers on full training set...")
        train_x = train[columns_to_use]
        train_y = train['purchased']

        if self.use_groups is None:
            for ranker in self.rankers:
                ranker.fit(train_x, train_y)

        else:
            train_baskets = train.groupby(['week', 'customer_id'])['article_id'].count().values
            for i in range(len(self.rankers)):
                if self.use_groups[i]:
                    self.rankers[i].fit(train_x, train_y, group=train_baskets)
                else:
                    self.rankers[i].fit(train_x, train_y)


        return self

        



    def predict(self, test_x, weighting=None):
        '''
        Ranks each of the given samples according to the weighted average score given by each of the base rankers.

        Parameters
            test_x (pandas.dataframe): dataframe of samples to predict on.
            weighting (string or None): metric (computed on validation during training) which rankers rankings are weighted by.

        Returns
            The array of average rankings where each ranking corresponds to the sample at the same index in test_x.
        '''
    
        rankings = np.zeros(len(test_x.index))

        for i in range(len(self.rankers)):

            if weighting == None:
                #just produce list of unweighted average ranks taken across each ranker (sum rank aggregation)
                rankings += rankdata(self.rankers[i].predict(test_x))


            else:
                #add weighted rankings where the weight is the specified by the weighting argument (weights were computed for each ranker on the validation set)
                rankings += (rankdata(self.rankers[i].predict(test_x)) * self.ranker_weights[weighting][i])        
        
        return rankdata(rankings)


        