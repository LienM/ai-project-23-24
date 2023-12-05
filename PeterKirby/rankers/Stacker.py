import numpy as np
from scipy.stats import rankdata
from Evaluations import get_preds_and_actual, mrr, mapk, precision_recall_at_k
import pandas as pd
from rankers.Ranker import Ranker


class Stacker():
    '''
    Class for ensembling rankers - sklearn has stacking class for classifiers and regressors but cannot pass rankers as base estimators.
    
    Each base estimator is trained on the whole dataset, then final ranker is trained on cross-validated predictions (as with sklearn stackers).
    '''

    def __init__(self, rankers, meta_model, use_groups=None, split_category=None, category_baskets=None):
        '''
        Parameters:
            rankers: a list of instances of base rankers that are used to create stacked ranker.
            meta_model: an instance of the model used to combine results of the weak rankers.
            use_groups (list or None): a list of booleans indicating which rankers should have data split into groups. If none, none of the rankers will.
        '''

        self.rankers = rankers
        self.ranker_weights = {
            "MRR": [],
            "MAPk": [],
            "Pk": []
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
            bestsellers_previous_week (None or pandas.DataFrame): frame of the bestsellers from the previous week
        
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
            
            train_x = train_no_val[[f"train{ranker_no}" for ranker_no in range(len(self.rankers))]].to_numpy()

            print(f"metamodel training shape: {train_x.shape}")

            train_y = train_no_val['purchased'].to_list()

            self.meta_model.fit(train_x, train_y)



        if compute_scores:
            #if we want to compute scores as well, compute weights as different metrics from performance on validation set.

            #1) train all rankers on all but last week of TRAINING data.
            #2) using these rankers, rank candidates for last week of TRAINING data - record MRR, MAP@k, precision@k.
            #3) retrain rankers on full training set


            print("Computing scores on validatation...")
            self.ranker_weights = {
                "MRR": [],
                "MAPk": [],
                "Pk": []
            }

            train_transactions = train_no_val[train_no_val['purchased'] == 1]
            for i in range(len(self.rankers)):
                #validation[f"ranker_{i}"] = self.rankers[i].predict(val_x)
                #look at what was actually bought in the validation week, make list of actual purchases, and get scores
                preds, actual_purchases, preds_dense, actual_purchases_dense = get_preds_and_actual(train_no_val, f"ranker{i}", train_transactions, pd.read_csv('../../../Data/sample_submission.csv'))         #doesnt use bestsellers_previous_week to pad predictions (these are made from candidate generation, not a result of ranker)

                self.ranker_weights["MRR"].append(mrr(actual_purchases_dense, preds_dense))
                self.ranker_weights["MAPk"].append(mapk(actual_purchases_dense, preds_dense, k=12))
                self.ranker_weights["Pk"].append(precision_recall_at_k(actual_purchases_dense, preds_dense, k=12)[0])




        #retraining base rankers
        print("retraining base rankers on full training set...")
        self._train_base_rankers(train_no_val, columns_to_use)

        return self

        



    def predict(self, test_x, columns_to_use, weighting=None):
        '''
        Ranks each of the given samples according to the weighted average score given by each of the base rankers.

        Parameters
            test_x (pandas.dataframe): dataframe of samples to predict on.
            weighting (string or None): metric (computed on validation during training) which rankers rankings are weighted by.

        Returns
            The array of average rankings where each ranking corresponds to the sample at the same index in test_x.
        '''
    


        if weighting == "metamodel":
            #Make predictions with base models, pass these through metamodel, then return metamodel output.
            print("Predicting with metamodel")
            #rank_matrix = np.empty((len(test_x.index), len(self.rankers)))
            for i in range(len(self.rankers)):
                #rank_matrix[:,i] = self.rankers[i].predict(test_x)

                #setting rankings to be per customer (rather than rankings of the entire test set) so input to metamodel is similar ranges to training data
                test_x[f"ranker{i}"] = self.rankers[i].predict(test_x[columns_to_use])
                test_x[f"ranker{i}"] = test_x.groupby(['week','customer_id'])[f"ranker{i}"].rank(ascending=False)


            rank_matrix = test_x[[f"ranker{ranker_no}" for ranker_no in range(len(self.rankers))]].to_numpy()

            print(f"Prediction matrix shape: {rank_matrix.shape}")
            print(f"prediction matrix:\n{rank_matrix}")

            return self.meta_model.predict(rank_matrix)



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
        for ranker_no in range(len(self.rankers)):
            #if split_category is specified, each ranker is trained on samples matching different values for the given category
            if self.split_category is not None:
                train_subset = train[train[self.split_category].isin(self.category_baskets[ranker_no])]
            else:
                train_subset = train

            train_x = train_subset[columns_to_use]
            train_y = train_subset['purchased']

            #print(f"ranker {ranker_no} subset head:\n{train_subset.head()}")
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
            nr_validation_weeks (int): the number of weeks used to train per validation split. E.g. if validation weeks was 2, then predictions for week 100 would be generated by model trained on week 99 and 98.
        
        '''

        print("computing validation predictions for each of the base rankers...")

        #for each week from minweek+nr_validation_weeks to maxweek
        min_week = train.week.min()
        max_week = train.week.max()


        for ranker_no in range(len(self.rankers)):
            train[f"ranker{ranker_no}"] = np.NaN
            for test_fold_week in range(min_week+nr_validation_weeks, max_week+1):

                #fitting to validation training set
                train_folds = train[(train.week < test_fold_week) & (train.week >= test_fold_week-nr_validation_weeks)]

                #if split_category is specified, each ranker is trained on samples matching different values for the given category
                if self.split_category is not None:
                    train_folds = train_folds[train_folds[self.split_category].isin(self.category_baskets[ranker_no])]

                    #print(f"Train folds head: {train_folds.head}")
                #may need to insert some caveat here that gives default value for current test set if the current training set has no values.

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


            '''
            if isinstance(self.meta_model, Ranker):
            #only if metamodel is a ranker: scale each set of rankings to between 0 (lowest ranking) and 1 (highest ranking) - evenly spaced
                train.groupby(['week', 'customer_id'], group_keys=False)[f"ranker{ranker_no}"].apply(lambda x: (x-min(x))/(max(x)-min(x)))
            '''


            '''
            #for each sample, change to sum of negative samples for that user for that week that are smaller than sample / sum of negative samples for that user for that week

            train[f"ranker{ranker_no}"] = train.apply(lambda x:
                                                      len(train[(train['week'] == x['week']) &
                                                                (train['customer_id'] == x['customer_id']) &
                                                                (train['purchased'] == 1) &
                                                                (train[f"ranker{ranker_no}"] > x[f"ranker{ranker_no}"])]) / 
                                                                len(train[(train['week'] == x['week']) &
                                                                (train['customer_id'] == x['customer_id']) &
                                                                (train['purchased'] == 1)]))
            '''


        return train
    



    '''
    def convert_scores(train):
        #method for converting rankings into 0,1 range by calculating proportion of negative samples that are ranked lower

        #TODO maybe insert this after weeks rankings have been predicted in _compute_validation_predictions (so that they are already grouped)
        #within each week: sum of negative samples whos ranking is lower than current sample / sum of all negative samples


        #for each week, for each customer
        #   get selection of predictions for that week
        #   count number of negative samples for that week
        #   for each prediction in selection

        train.groupby(['week', 'customer_id'])
        



        train.apply()

    '''



        