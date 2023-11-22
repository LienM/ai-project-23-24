import numpy as np
from sklearn.base import clone
from scipy.stats import rankdata
from rankers.Ranker import Ranker

class Bagger():
    '''
    Class for ensembling rankers - sklearn has bagging class for classifiers and regressors but cannot pass rankers as base estimators.
    
    How it works:
    1) split data into random (overlapping) subsets - proportion of full dataset is given as an argument.
    2) train one classifier on each subset
    3) when predicting: combine results by taking the average value for each prediction
    
    '''
    def __init__(self, ranker, nr_rankers, samples_proportion):
        '''
        Parameters
            ranker (object): initialised ranker used as the base ranker (will be retrained multiple times). Must have fit() and predict() functions.
            nr_rankers (int): the number of base rankers to use for the ensemble.
            samples_proportion (float): the proportion of training samples used to train each base ranker.
        '''
        self.base_ranker = ranker
        self.nr_rankers = nr_rankers
        self.samples_proportion = samples_proportion
        self.rankers = []



    def fit(self, train, columns_to_use, use_groups=False):
        '''
        Fits rankers to (overlapping) subsets of the training data.

        Parameters
            train (pd.dataframe): The dataframe of training data - each row is a sample, each column represents a feature of the data.
            columns_to_use (list): list of strings of column headers which are to be used to train the rankers.
            use_groups (boolean): if each dataset is further split into groups for training a single base ranker (e.g. for LGBM ranker), then set to True. Default is False.

        Returns
            This Bagger object.
        '''

        self.rankers = []                   #reinitialising self.rankers to empty array
        for e in range(self.nr_rankers):
            samples = train.sample(frac=self.samples_proportion)
            if use_groups:
                samples_baskets = samples.groupby(['week', 'customer_id'])['article_id'].count().values

            samples_x = samples[columns_to_use]
            samples_y = samples['purchased']
            
            #small workaround for using different types of rankers (wrapped sklearn classifiers or LGBM ranker)
            if isinstance(self.base_ranker, Ranker):
                ranker_clone = self.base_ranker.clone()
            else:
                ranker_clone = clone(self.base_ranker)


            if use_groups:
                self.rankers.append(
                    ranker_clone.fit(
                        samples_x,
                        samples_y,
                        group=samples_baskets
                        )
                    )
            else:
                self.rankers.append(
                    ranker_clone.fit(
                        samples_x,
                        samples_y
                        )
                    )
                
        return self
    
    def predict(self, test_x):
        '''
        Ranks each of the given samples according to the average score given by each of the base rankers.

        Parameters
            test_x (pandas.dataframe): dataframe of samples to predict on.

        Returns
            The array of average rankings where each ranking corresponds to the sample at the same index in test_x.
        '''

        preds = np.zeros(len(test_x.index))
        for i in range(self.nr_rankers):
            preds += self.rankers[i].predict(test_x)
        preds /= self.nr_rankers

        return rankdata(preds)

