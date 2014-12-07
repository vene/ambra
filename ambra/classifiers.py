from sklearn.linear_model import LogisticRegression

from pairwise import pairwise_transform, flip_pairs

class IntervalLogisticRegression(LogisticRegression):
    def fit(self, X, y):
        X_pw, _ = pairwise_transform(X, y)
        X_pw, y_pw = flip_pairs(X_pw, random_state=0)  # not fair
        return super(IntervalLogisticRegression, self).fit(X_pw, y_pw)

    def score(self, X, y):
        print("pairwise accuracy is used")
        X_pw, _ = pairwise_transform(X, y)
        X_pw, y_pw = flip_pairs(X_pw, random_state=0)  # not fair
        return super(IntervalLogisticRegression, self).score(X_pw, y_pw)


if __name__ == '__main__':
    import numpy as np
    X = np.arange(10)[:, np.newaxis]
    Y = [[4, 7], [1, 3], [2, 4], [8, 15], [5, 6], [1, 2], [10, 11],
         [10, 12], [10, 13], [10, 14]]
    from sklearn.cross_validation import KFold, cross_val_score
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y, random_state=0)
    print cross_val_score(IntervalLogisticRegression(C=1.0),
                          X, Y, cv=KFold(len(X), n_folds=3))
