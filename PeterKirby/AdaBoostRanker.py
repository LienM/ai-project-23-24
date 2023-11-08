from sklearn.ensemble import AdaBoostClassifier

class AdaBoostRanker():
    '''
    Wrapper class to do fitting/prediction using sklearn adaboost classifier, but returning ranked list.
    '''

    def __init__(self):
        self.ranker = AdaBoostClassifier()
        self.feature_importances_ = None

    def fit(self, x, y):
        '''
            Fits the training data to the classifier and returns this AdaBoostRanker object.
        '''
        self.ranker = self.ranker.fit(x,y)
        self.feature_importances_ = self.ranker.feature_importances_
        return self

    def predict(self, x):
        '''
            Ranking is just done by the confidence of the ranker.
        '''
        predictions = self.ranker.predict_log_proba(x)
        return predictions
