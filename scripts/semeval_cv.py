import sys
import json
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.pipeline import make_pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle

with open(sys.argv[1]) as f:
    entries = json.load(f)

X_len = np.array([len(doc['text'].split()) for doc in entries])[:, np.newaxis]
X_txt = np.array([doc['text'] for doc in entries])
Y = np.array([doc['true_interval'] for doc in entries])
Y_possible = np.array([doc['possible_intervals'] for doc in entries])

X_len, X_txt, Y, Y_possible = shuffle(X_len, X_txt, Y, Y_possible,
                                      random_state=0)


from ambra.classifiers import IntervalLogisticRegression, interval_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_df=0.95, max_features=50, use_idf=False, norm="l1")
pipe = make_pipeline(vect, IntervalLogisticRegression(n_neighbors=100))

grid = GridSearchCV(IntervalLogisticRegression(C=1.0),
                    dict(n_neighbors=(5, 50, 100)),
                    verbose=True, cv=KFold(len(X_len), n_folds=5),
                    scoring=interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible))

grid.fit(X_len, Y)
print grid.best_score_, grid.best_params_

grid = GridSearchCV(pipe, dict(intervallogisticregression__C=[1.0]),
                    verbose=True, cv=KFold(len(X_len), n_folds=5),
                    scoring=interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible))

grid.fit(X_txt, Y)
print grid.best_score_, grid.best_params_
