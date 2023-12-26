import numpy as np
from scipy.stats import rankdata
from Evaluations import get_preds_and_actual, mrr, mapk
import pandas as pd


class Stacker():
    '''
    Class for ensembling rankers - sklearn has stacking class for classifiers and regressors but cannot pass rankers as base estimators.
    
    Each base estimator is trained on the whole dataset, then final ranker is trained on cross-validated predictions (as with sklearn stackers).
    '''

    def __init__(self, rankers, meta_model, use_groups=None, split_category=None, category_baskets=None):
        '''
        Parameters:
            rankers (list of rankers): a list of instances of base rankers that are used to create stacked ranker.
            meta_model (model that implements train and predict): an instance of the model used to combine results of the weak rankers.
            use_groups (list or None): a list of booleans indicating which rankers should have data split into groups. If none, none of the rankers will.
            split_category (string or None): string representing the feature that samples are to be split by when training base rankers on different data splits (is using bagger/stacker hybrid). None if each ranker is trained on the full dataset.
            category_baskets (list or None): list of lists where each list represents the values of split_category that are used for the nth ranker. E.g. if category_baskets = [[1,2], [3], [4, 5]], then first ranker is traned on samples where split_category == 1 or 2, ranker 2 on 3, and ranker3 on 4 or 5
        '''

        self.rankers = rankers
        self.ranker_weights = {
            "MRR": [],
            "MAPk": [],
        }

        self.meta_model = meta_model

        self.use_groups = use_groups

        self.split_category = split_category

        self.category_baskets = category_baskets
        


    def fit(self, train, columns_to_use, nr_validation_weeks, compute_scores=True):
        '''
        Computes ranker weights by splitting training set into training/validation and judging performance on validation set, then fits all of the base rankers to the whole dataset.
               
        Parameters
            train (pd.dataframe): The dataframe of training data - each row is a sample, each column represents a feature of the data.
            columns_to_use (list): list of strings of column headers which are to be used to train the rankers.
            nr_validation_weeks (int): the number of weeks used only for computing validation scores on future weeks (not used for final training step of retraining base rankers)
            compute_scores (boolean): if true, compute the MRR and MAP@12 scores on training weeks

        Returns:
            Stacker: the trained model
        '''

        #computing predictions of validation
        if self.split_category is not None and self.category_baskets is None:
            #fill category baskets with n lists containing which values for the given category each of the n rankers should be trained on
            vals = train[self.split_category].unique()
            
            #if baskets arent given, shuffle list - if not shuffled, splits on different categories with same number of base rankers will give same training baskets
            np.random.shuffle(vals)

            #if there are more category options than rankers, split categories into chunks of (close to) equal size, assigning each chunk to a ranker. Chunks are non-overlapping.
            if len(vals) > len(self.rankers):
                self.category_baskets = np.array_split(vals, len(self.rankers))

            #if there are more rankers than categories, then randomly assign one category to each ranker (each category represented by at least one ranker)
            else:
                self.category_baskets = []
                for i in range(len(self.rankers)):
                    self.category_baskets.append([vals[i%len(vals)]])

            print(f"Category '{self.split_category}' splits: {self.category_baskets}")


        self._compute_validation_predictions(train, columns_to_use, nr_validation_weeks)
        train_no_val = train[(train.week >= train.week.min()+nr_validation_weeks)]


        if self.meta_model is not None:
            #training metamodel (training data is the predictions from the rankers done during validation)
            print("training metamodel")
            for i in range(len(self.rankers)):
                train_no_val[f"train{i}"] = train.groupby(['week', 'customer_id'])[f"ranker{i}"].rank(ascending=False)              #ascending so "best rank" is always the same number (1) - same done when predicting
            
            train_x = train_no_val[[f"train{ranker_no}" for ranker_no in range(len(self.rankers))]].to_numpy()                      #isolating rankers' predictions for use as training data

            print(f"metamodel training shape: {train_x.shape}")

            train_y = train_no_val['purchased'].to_list()

            self.meta_model.fit(train_x, train_y)


        if compute_scores:
            #if we want to compute scores as well, compute weights as different metrics from performance on validation set.

            print("Computing scores on validatation...")
            self.ranker_weights = {
                "MRR": [],
                "MAPk": [],
            }

            train_transactions = train_no_val[train_no_val['purchased'] == 1]
            for i in range(len(self.rankers)):
                #look at what was actually bought in the validation week, make list of actual purchases, and get scores
                preds, actual_purchases, preds_dense, actual_purchases_dense = get_preds_and_actual(train_no_val, f"ranker{i}", train_transactions, pd.read_csv('../../../Data/sample_submission.csv'))         #doesnt use bestsellers_previous_week to pad predictions (these are made from candidate generation, not a result of ranker)

                self.ranker_weights["MRR"].append(mrr(actual_purchases_dense, preds_dense))
                self.ranker_weights["MAPk"].append(mapk(actual_purchases_dense, preds_dense, k=12))


        #retraining base rankers
        print("retraining base rankers on full training set...")
        self._train_base_rankers(train_no_val, columns_to_use)
        

        return self

        
    def predict(self, test_x, columns_to_use, weighting=None):
        '''
        Ranks each of the given samples according to the weighted average score given by each of the base rankers.

        Parameters
            test_x (pandas.dataframe): dataframe of samples to predict on.
            columns_to_use (list): the list of columns considered when making predictions
            weighting (string or None): metric (computed on validation during training) which rankers rankings are weighted by.

        Returns
            The array of average rankings where each ranking corresponds to the sample at the same index in test_x.
        '''
    
        if weighting == "metamodel":
            #Make predictions with base models, pass these through metamodel, then return metamodel output.
            print("Predicting with metamodel")
            
            for i in range(len(self.rankers)):
                #setting rankings to be per customer (rather than rankings of the entire test set) so input to metamodel is similar ranges to training data
                test_x[f"ranker{i}"] = self.rankers[i].predict(test_x[columns_to_use])
                test_x[f"ranker{i}"] = test_x.groupby(['week','customer_id'])[f"ranker{i}"].rank(ascending=False)


            rank_matrix = test_x[[f"ranker{ranker_no}" for ranker_no in range(len(self.rankers))]].to_numpy()

            print(f"Prediction matrix shape: {rank_matrix.shape}")
            print(f"prediction matrix:\n{rank_matrix}")

            return self.meta_model.predict(rank_matrix)


        #if weighting is not using metamodel, then just compute weighted average rank across all rankers' rankings.
        print(f"Predicting with {weighting} weighting")

        rankings = np.zeros(len(test_x.index))

        for i in range(len(self.rankers)):
            if weighting == None:
                #just produce list of unweighted average ranks taken across each ranker (sum rank aggregation)
                rankings += rankdata(self.rankers[i].predict(test_x[columns_to_use]))

            else:
                #add weighted rankings where the weight is the specified by the weighting argument (weights were computed for each ranker on the validation set)
                rankings += (rankdata(self.rankers[i].predict(test_x[columns_to_use])) * self.ranker_weights[weighting][i])        
        
        return rankdata(rankings)



    def _train_base_rankers(self, train, columns_to_use):
        '''
        Internal function used for training the base rankers on the training data using the given columns in columns_to_use
        '''
        for ranker_no in range(len(self.rankers)):
            #if split_category is specified (for bagger/stacker hybrid), each ranker is trained on samples matching different values for the given category
            if self.split_category is not None:
                train_subset = train[train[self.split_category].isin(self.category_baskets[ranker_no])]
            else:
                train_subset = train

            train_x = train_subset[columns_to_use]
            train_y = train_subset['purchased']

            if self.use_groups is None:
                self.rankers[ranker_no].fit(train_x, train_y)
            
            else:
                train_baskets = train_subset.groupby(['week', 'customer_id'])['article_id'].count().values

                if self.use_groups[ranker_no]:
                    self.rankers[ranker_no].fit(train_x, train_y, group=train_baskets)

                else:
                    self.rankers[ranker_no].fit(train_x, train_y)





    def _compute_validation_predictions(self, train, columns_to_use, nr_validation_weeks=10):
        '''
        Computes validation predictions on each week of the test set > nr_validation_weeks.
        The validation predictions are added to the train dataframe as columns - one for each ranker

        Parameters:
            train (pandas.DataFrame): te dataframe containing the data samples used for training
            columns_to_use (list): the list of columns considered when training/predicting
            nr_validation_weeks (int): the number of weeks used to train per validation split. E.g. if validation weeks was 2, then predictions for week 100 would be generated by model trained on week 99 and 98.
        
        Returns:
            train (pandas.DataFrame): the original dataframe also with columns added with the ranking predictions made by base rankers (each base ranker has its own column containing that rankers' rankings)
        '''

        print("computing validation predictions for each of the base rankers...")

        #for each week from minweek+nr_validation_weeks to maxweek
        min_week = train.week.min()
        max_week = train.week.max()


        for ranker_no in range(len(self.rankers)):
            train[f"ranker{ranker_no}"] = np.NaN


            #1) train ranker of nr_validation_weeks training weeks,
            #2) compute rnakings for the next training week
            #3) repeat for each week that has 5 prior weeks available
            for test_fold_week in range(min_week+nr_validation_weeks, max_week+1):

                #fitting to validation training set
                train_folds = train[(train.week < test_fold_week) & (train.week >= test_fold_week-nr_validation_weeks)]

                #if split_category is specified, each ranker is trained on samples matching different values for the given category
                if self.split_category is not None:
                    train_folds = train_folds[train_folds[self.split_category].isin(self.category_baskets[ranker_no])]

                #in future, may need to insert some caveat here that gives default value for current test set if the current training set has no values.

                train_x = train_folds[columns_to_use]
                train_y = train_folds['purchased']


                if self.use_groups is not None:
                    baskets = train_folds.groupby(['week', 'customer_id'])['article_id'].count().values
                    if self.use_groups[ranker_no]:
                        self.rankers[ranker_no].fit(train_x, train_y, group=baskets)

                    else:
                        self.rankers[ranker_no].fit(train_x, train_y)


                else:
                    self.rankers[ranker_no].fit(train_x, train_y)



                #predicting on validation fold, adding validation scores to the train dataframe

                test_fold = train[train.week == test_fold_week]

                test_x = test_fold[columns_to_use]


                preds = self.rankers[ranker_no].predict(test_x)


                #filling in the appropriate rows with the predictions from that ranker
                train.loc[train['week'] == test_fold_week, [f"ranker{ranker_no}"]] = preds


            train[f"ranker{ranker_no}"] = train.groupby(['week', 'customer_id'])[f"ranker{ranker_no}"].rank()


        return train
    
        