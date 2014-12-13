""" Scoring functions for interval spaces """

# Author: Vlad Niculae <vn66@cornell.edu>
# Author: Alina Ciobanu <alinaciobanu20@gmail.com>
# License: Simplified BSD

import numpy as np

def mae_interval_score(Y_pred, Y_true):
    """Mean absolute error between interval centers"""
    return np.mean(np.abs((Y_true - Y_pred).mean(axis=1)))


def mae_interval_scorer(est, X, Y, Y_possible):
    pred = est.predict(X, Y_possible)
    return -mae_interval_score(pred, Y)



SEMEVAL_GRID = [0.0, 0.1, 0.15, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99]


def _semeval_score_one(Y_pred, Y_true, Y_possible):
    true_idx = Y_possible.index(list(Y_true))
    pred_idx = Y_possible.index(list(Y_pred))
    dist = np.abs(true_idx - pred_idx)
    if dist >= len(SEMEVAL_GRID):
        return SEMEVAL_GRID[-1]
    else:
        return SEMEVAL_GRID[dist]


def semeval_interval_score(Y_pred, Y_true, Y_possible):
    """Semeval manually defined scoring grid"""
    return np.mean([_semeval_score_one(y_pred, y_true, y_possible)
                    for y_pred, y_true, y_possible
                    in zip(Y_pred, Y_true, Y_possible)])


def semeval_interval_scorer(est, X, Y, Y_possible):
    pred = est.predict(X, Y_possible)
    return -semeval_interval_score(pred, Y, Y_possible)
