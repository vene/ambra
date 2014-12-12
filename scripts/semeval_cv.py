import sys
import json
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle


class MyPipeline(Pipeline):
    def predict(self, X, Y_possible):
            """Applies transforms to the data, and the predict method of the
            final estimator. Valid only if the final estimator implements
            predict."""
            Xt = X
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)
            return self.steps[-1][-1].predict(Xt, Y_possible)


with open(sys.argv[1]) as f:
    entries = json.load(f)

X_len = np.array([len(doc['text'].split()) for doc in entries])[:, np.newaxis]
X_txt = np.array([doc['text'] for doc in entries])
Y = np.array([doc['true_interval'] for doc in entries])
Y_possible = np.array([doc['possible_intervals'] for doc in entries])

X_len, X_txt, Y, Y_possible = shuffle(X_len, X_txt, Y, Y_possible,
                                      random_state=0)

# make it run fast
X_len = X_len[:100]
X_txt = X_txt[:100]
Y = Y[:100]
Y_possible = Y_possible[:100]

from ambra.classifiers import IntervalLogisticRegression, interval_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

grid = GridSearchCV(IntervalLogisticRegression(C=1.0),
                    dict(n_neighbors=(5, 50, 100)),
                    verbose=True, cv=KFold(len(X_len), n_folds=5),
                    scoring=interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible))

grid.fit(X_len, Y)
print grid.best_score_, grid.best_params_

vect = TfidfVectorizer(max_df=0.95, max_features=50, use_idf=False, norm="l1")
pipe = MyPipeline([('vect', vect),
                   ('clf', IntervalLogisticRegression(n_neighbors=100))])


grid = GridSearchCV(pipe, dict(clf__C=[0.01, 1.0, 10.0]),
                    verbose=True, cv=KFold(len(X_len), n_folds=5),
                    scoring=interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible))

grid.fit(X_txt, Y)
print grid.best_score_, grid.best_params_
