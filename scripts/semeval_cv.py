import sys
import json
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.utils import shuffle

with open(sys.argv[1]) as f:
    entries = json.load(f)

X = np.array([len(entry['text'].split()) for entry in entries])
X = X[:, np.newaxis]
Y = np.array([entry['true_interval'] for entry in entries])
X, Y = shuffle(X, Y, random_state=0)


print X.shape, Y.shape

from ambra.classifiers import IntervalLogisticRegression
print cross_val_score(IntervalLogisticRegression(C=1.0),
                      X, Y, cv=KFold(len(X), n_folds=5))

