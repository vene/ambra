from __future__ import print_function
import sys
import json
import numpy as np

from scipy.stats import sem

from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer

from ambra.cross_validation import cross_val_score
from ambra.tools import PossiblePipeline, Proj, IntervalSelectKBest
from ambra.features import LengthFeatures, StylisticFeatures
from ambra.features import NgramLolAnalyzer
from ambra.interval_scoring import semeval_interval_scorer
from ambra.classifiers import IntervalLogisticRegression

fname = sys.argv[1]
with open(fname) as f:
        entries = json.load(f)

# some buggy docs are empty
entries = [entry for entry in entries if len(entry['lemmas'])]

X = np.array(entries)
Y = np.array([doc['interval'] for doc in entries])
Y_possible = np.array([doc['all_fine_intervals'] for doc in entries])

X, Y, Y_possible = shuffle(X, Y, Y_possible, random_state=0)

print("Length features")
print("===============")
pipe = PossiblePipeline([('vect', Proj(LengthFeatures(), key='lemmas')),
                         ('scale', StandardScaler(with_mean=False,
                                                  with_std=True)),
                         ('clf', IntervalLogisticRegression(C=0.0008030857221,
                                                            n_neighbors=10,
                                                            limit_pairs=1,
                                                            random_state=0))])

scores = cross_val_score(pipe, X, Y, cv=KFold(len(X), n_folds=5),
                         scoring=semeval_interval_scorer,
                         scorer_params=dict(Y_possible=Y_possible),
                         n_jobs=4)
print("{:.3f} +/- {:.4f}".format(scores.mean(), sem(scores)))

print()
print("Stylistic features")
print("==================")

union = FeatureUnion([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                      ('style', StylisticFeatures())])

pipe = PossiblePipeline([('vect', union),
                         ('scale', StandardScaler(with_mean=False,
                                                  with_std=True)),
                         ('clf', IntervalLogisticRegression(C=0.02154434690032,
                                                            n_neighbors=10,
                                                            limit_pairs=1,
                                                            random_state=0))])

scores = cross_val_score(pipe, X, Y, cv=KFold(len(X), n_folds=5),
                         scoring=semeval_interval_scorer,
                         scorer_params=dict(Y_possible=Y_possible),
                         n_jobs=4)
print("{:.3f} +/- {:.4f}".format(scores.mean(), sem(scores)))


print()
print("Full")
print("====")

vectorizer = TfidfVectorizer(use_idf=False, norm='l1',
                             analyzer=NgramLolAnalyzer(lower=False))
vectorizer_low = TfidfVectorizer(use_idf=False, norm='l1',
                                 analyzer=NgramLolAnalyzer(lower=True))

union = FeatureUnion([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                      ('style', StylisticFeatures()),
                      ('pos', Proj(clone(vectorizer), key='pos')),
                      ('tokens', Proj(clone(vectorizer_low), key='tokens'))])

final_pipe = PossiblePipeline([('union', union),
                               ('scale', StandardScaler(with_mean=False,
                                                        with_std=True)),
                               ('fs', IntervalSelectKBest(chi2)),
                               ('clf', IntervalLogisticRegression(
                                    n_neighbors=10,
                                    limit_pairs=0.01,  # make larger if possible
                                    random_state=0))])

final_pipe.set_params(**{'union__tokens__transf__min_df': 5,
                         'union__tokens__transf__max_df': 0.9,
                         'union__pos__transf__analyzer__ngram_range': (2, 2),
                         'union__pos__transf__max_df': 0.8,
                         'fs__k': 2000,
                         'union__pos__transf__min_df': 1,
                         'clf__C': 2.592943797404667e-05,
                         'union__tokens__transf__analyzer__ngram_range': (1, 1)}
)

scores = cross_val_score(final_pipe, X, Y, cv=KFold(len(X), n_folds=5),
                         scoring=semeval_interval_scorer,
                         scorer_params=dict(Y_possible=Y_possible),
                         n_jobs=4
)

print("{:.3f} +/- {:.4f}".format(scores.mean(), sem(scores)))

final_pipe.fit(X, Y)
feature_names = final_pipe.steps[0][1].get_feature_names()
feature_names = np.array(feature_names)[final_pipe.steps[2][1].get_support()]

coef = final_pipe.steps[-1][-1].coef_.ravel()

for idx in np.argsort(-np.abs(coef))[:100]:
    print("{:.2f}\t{}".format(coef[idx], feature_names[idx]))
