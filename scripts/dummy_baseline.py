from __future__ import print_function
import sys
import json
import numpy as np

from scipy.stats import sem

from sklearn.cross_validation import KFold
from ambra.cross_validation import cross_val_score
from sklearn.utils import shuffle
from ambra.classifiers import DummyIntervalClassifier

from ambra.interval_scoring import semeval_interval_scorer

with open(sys.argv[1]) as f:
    entries = json.load(f)

# some buggy docs are empty
entries = [entry for entry in entries if len(entry['lemmas'])]

X = np.array(entries)
Y = np.array([doc['interval'] for doc in entries])
Y_possible = np.array([doc['all_fine_intervals'] for doc in entries])

X, Y, Y_possible = shuffle(X, Y, Y_possible, random_state=0)

for method in ["center", "random"]:
    dummy_scores = cross_val_score(DummyIntervalClassifier(method=method,
                                                           random_state=0),
                                   X, Y, cv=KFold(len(X), n_folds=5),
                                   scoring=semeval_interval_scorer,
                                   scorer_params=dict(Y_possible=Y_possible))
    dummy_scores = 1 + dummy_scores
    print("{} \t {:.3f} \pm {:.3f}".format(method,
                                           np.mean(dummy_scores),
                                           sem(dummy_scores)))
