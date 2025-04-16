# offense_classifier/ensemble.py
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Soft voting entre plusieurs modèles sklearn
class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        self.estimators = estimators
        self.model = VotingClassifier(estimators=estimators, voting='soft')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Exemple d'utilisation
# ensemble = SoftVotingEnsemble([
#     ('rf', RFModel().model),
#     ('svm', SVMModel().model)
# ])