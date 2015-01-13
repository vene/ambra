import numbers
import time

import numpy as np

from sklearn.utils import safe_indexing
from sklearn.base import is_classifier, clone
from sklearn.metrics.scorer import check_scoring
from sklearn.externals.joblib import Parallel, delayed, logger
from ambra.backports import _num_samples, indexable
from sklearn.cross_validation import check_cv

def _safe_split(estimator, X, y, indices, train_indices=None):
    """Create subset of dataset and properly handle kernels."""
    if hasattr(estimator, 'kernel') and callable(estimator.kernel):
        # cannot compute the kernel values with custom function
        raise ValueError("Cannot use a custom kernel function. "
                         "Precompute the kernel matrix instead.")

    if not hasattr(X, "shape"):
        if getattr(estimator, "_pairwise", False):
            raise ValueError("Precomputed kernels or affinity matrices have "
                             "to be passed as arrays or sparse matrices.")
        X_subset = [X[idx] for idx in indices]
    else:
        if getattr(estimator, "_pairwise", False):
            # X is a precomputed square kernel matrix
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square kernel matrix")
            if train_indices is None:
                X_subset = X[np.ix_(indices, indices)]
            else:
                X_subset = X[np.ix_(indices, train_indices)]
        else:
            X_subset = safe_indexing(X, indices)

    if y is not None:
        y_subset = safe_indexing(y, indices)
    else:
        y_subset = None

    return X_subset, y_subset


def _score(estimator, X_test, y_test, scorer, **params):
    """Compute the score of an estimator on a given test set."""
    if y_test is None:
        score = scorer(estimator, X_test, **params)
    else:
        score = scorer(estimator, X_test, y_test, **params)
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                         % (str(score), type(score)))
    return score


def cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, pre_dispatch='2*n_jobs',
                    scorer_params=None):
    """Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : cross-validation generator or int, optional, default: None
        A cross-validation generator to use. If int, determines
        the number of folds in StratifiedKFold if y is binary
        or multiclass and estimator is a classifier, or the number
        of folds in KFold otherwise. If None, it is equivalent to cv=3.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    scorer_params : dict, optional
        Parameters to pass to the scorer.  Can be used for sample weights
        and sample groups.

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    X, y = indexable(X, y)

    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(estimator), X, y,
                                              scorer, train, test, verbose,
                                              None, fit_params, scorer_params)
                      for train, test in cv)
    return np.array(scores)[:, 0]


def _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters,
                   fit_params, scorer_params, return_train_score=False,
                   return_parameters=False):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like or None
        The target variable to try to predict in the case of
        supervised learning.

    scoring : callable
        A scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape = (n_train_samples,)
        Indices of training samples.

    test : array-like, shape = (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    scorer_params : dict or None
        Parameters that will be passed to the scorer.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    Returns
    -------
    train_score : float, optional
        Score on training set, returned only if `return_train_score` is `True`.

    test_score : float
        Score on test set.

    n_test_samples : int
        Number of test samples.

    scoring_time : float
        Time spent for fitting and scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.
    """
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                          for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust lenght of sample weights
    n_samples = _num_samples(X)
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, np.asarray(v)[train]
                       if hasattr(v, '__len__') and len(v) == n_samples else v)
                       for k, v in fit_params.items()])

    # Same, but take both slices
    scorer_params = scorer_params if scorer_params is not None else {}
    train_scorer_params = dict([(k, np.asarray(v)[train]
                                 if hasattr(v, '__len__')
                                 and len(v) == n_samples
                                 else v)
                                for k, v in scorer_params.items()])
    test_scorer_params = dict([(k, np.asarray(v)[test]
                                if hasattr(v, '__len__')
                                and len(v) == n_samples
                                else v)
                               for k, v in scorer_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    if y_train is None:
        estimator.fit(X_train, **fit_params)
    else:
        estimator.fit(X_train, y_train, **fit_params)
    test_score = _score(estimator, X_test, y_test, scorer,
                        **test_scorer_params)
    if return_train_score:
        train_score = _score(estimator, X_train, y_train, scorer,
                             **train_scorer_params)

    scoring_time = time.time() - start_time

    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(X_test), scoring_time])
    if return_parameters:
        ret.append(parameters)
    return ret


