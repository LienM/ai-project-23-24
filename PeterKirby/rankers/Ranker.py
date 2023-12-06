from sklearn.base import clone
from scipy.stats import rankdata



class Ranker():
    '''
    Wrapper class to do fitting/prediction using an sklearn classifier, but returning ranked list.        
    '''

    def __init__(self, ranker):
        self.ranker = ranker
        self.feature_importances_ = None

    def fit(self, x, y):
        '''
        Fits the training data to the classifier and returns this object.

        Parameters
            x: training features
            y: training labels

        Returns
            This ranker object.
        '''
        self.ranker = self.ranker.fit(x,y)
        if hasattr(self.ranker, "feature_importances_"):
            self.feature_importances_ = self.ranker.feature_importances_
        return self

    def predict(self, x):
        '''
        Predicts rankings for each input sample
        Ranking is just done by the confidence of the ranker so rankings arent ints, but their order specifies their ranking.

        Parameters
            x: samples to predict ranking of

        Returns
            The confidence values of the inputs (to rank, just order the confidence values)
        '''
        predictions = self.ranker.predict_log_proba(x)[:,1]
        return rankdata(predictions)
    

    def clone(self):
        return Ranker(clone(self.ranker))
