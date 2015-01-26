import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest

class PossiblePipeline(Pipeline):
    """Pipeline subclass that supports Y_possible in predict"""
    def predict(self, X, Y_possible):
            """Applies transforms to the data, and the predict method of the
            final estimator. Valid only if the final estimator implements
            predict."""
            Xt = X
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)
            return self.steps[-1][-1].predict(Xt, Y_possible)


class Proj(BaseEstimator, TransformerMixin):
    """Projection metaestimator for list-of-dict kind of data"""
    def __init__(self, transf, key):
        self.transf = transf
        self.key = key

    def fit(self, X, y=None):
        self.transf = self.transf.fit([x[self.key] for x in X], y)
        return self

    def transform(self, X):
        return self.transf.transform([x[self.key] for x in X])

    def fit_transform(self, X, y=None):
        return self.transf.fit_transform([x[self.key] for x in X], y)

    def get_feature_names(self):
        return self.transf.get_feature_names()


def _get_coarse_interval(year):
    year = int(year)
    lower = year / 100 * 100 + 50 * (year % 100 >= 50)
    return str(lower) + "-" + str(lower + 50)


class IntervalSelectKBest(SelectKBest):
    """Feature selection for interval labels by reducing to the center"""
    def fit(self, X, y):
        if self.k >= X.shape[1]:
            self.k = X.shape[1]
        y_flat = [_get_coarse_interval(np.mean(yelem)) for yelem in y]
        return super(IntervalSelectKBest, self).fit(X, y_flat)

