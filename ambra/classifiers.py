import numpy as np
from sklearn.linear_model import LogisticRegression

from pairwise import pairwise_transform, flip_pairs


from sklearn.linear_model import LogisticRegression

from ambra.pairwise import pairwise_transform, flip_pairs

def _nearest_sorted(scores, to_find, k=10):
    position = np.searchsorted(scores, to_find)
    width = k / 2
    offset = k % 2
    if position < width:
        return slice(None, k)
    elif position > len(scores) - width - offset:
        return slice(-k, None)
    else:
        return slice(position - width, position + width + offset)

def _interval_dist(a, b):
    a_lo, a_hi = a
    b_lo, b_hi = b
    if b_lo >= a_lo and b_hi <= a_hi:  # b contained in a
        return 0.0
    else:
        return np.abs(0.5 * (b_lo + b_hi - a_lo - a_hi))


class IntervalLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, n_neighbors=5):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X_pw, _ = pairwise_transform(X, y)
        X_pw, y_pw = flip_pairs(X_pw, random_state=0)  # not fair
        super(IntervalLogisticRegression, self).fit(X_pw, y_pw)
        train_scores = np.dot(X, self.coef_.ravel())
        order = np.argsort(train_scores)
        self.train_intervals_ = y[order]
        self.train_scores_ = train_scores[order]
        return self

    def score(self, X, y):
        print("pairwise accuracy is used")
        X_pw, _ = pairwise_transform(X, y)
        X_pw, y_pw = flip_pairs(X_pw, random_state=0)  # not fair
        return super(IntervalLogisticRegression, self).score(X_pw, y_pw)

    def _predict_interval(self, score, possible_intervals):
        interval_scores = [sum(_interval_dist(cand, nearest)
                               for nearest
                               in self.train_intervals_[
                                    _nearest_sorted(self.train_scores_,
                                                    score, k=self.n_neighbors)])
                           for cand in possible_intervals]
        return possible_intervals[np.argmin(interval_scores)]

    def predict(self, X, Y_possible):
        pred_scores = np.dot(X, self.coef_.ravel())
        return [self._predict_interval(score, possible_intervals)
                for score, possible_intervals
                in zip(pred_scores, Y_possible)]


def interval_scorer(est, X, Y, Y_possible):
    pred = est.predict(X, Y_possible)
    return -np.mean(np.abs((Y - pred)[:, 0]))


if __name__ == '__main__':
    X = np.arange(10)[:, np.newaxis]
    Y = [[4, 7], [1, 3], [2, 4], [8, 15], [5, 6], [1, 2], [10, 11],
         [10, 12], [10, 13], [10, 14]]
    from sklearn.cross_validation import KFold, cross_val_score
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y, random_state=0)
    print cross_val_score(IntervalLogisticRegression(C=1.0),
                          X, Y, cv=KFold(len(X), n_folds=3))
