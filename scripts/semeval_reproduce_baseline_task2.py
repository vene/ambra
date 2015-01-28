import sys
import json
import numpy as np

from scipy.stats import sem

from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler

from ambra.cross_validation import cross_val_score
from ambra.tools import PossiblePipeline, Proj
from ambra.features import LengthFeatures
from ambra.interval_scoring import semeval_interval_scorer
from ambra.classifiers import IntervalRidge

fname = sys.argv[1]
with open(fname) as f:
        entries = json.load(f)

# some buggy docs are empty
entries = [entry for entry in entries if len(entry['lemmas'])]

X = np.array(entries)
Y = np.array([doc['interval'] for doc in entries])
Y_possible = np.array([doc['all_fine_intervals'] for doc in entries])

X, Y, Y_possible = shuffle(X, Y, Y_possible, random_state=0)



pipe = PossiblePipeline([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                   ('scale', StandardScaler(with_mean=False, 
					    with_std=True)),
                   ('clf', IntervalRidge(max_iter=100, 
                                         alpha=0.1))])

scores = cross_val_score(pipe, X, Y, cv=KFold(len(X), n_folds=5),
                         scoring=semeval_interval_scorer,
                         scorer_params=dict(Y_possible=Y_possible),
                         n_jobs=4)

print("{:.3f} +/- {:.4f}".format(scores.mean(), sem(scores)))
