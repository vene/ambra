import sys
import json
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ambra.classifiers import IntervalLogisticRegression
from ambra.interval_scoring import semeval_interval_scorer

class MyPipeline(Pipeline):
    def predict(self, X, Y_possible):
            """Applies transforms to the data, and the predict method of the
            final estimator. Valid only if the final estimator implements
            predict."""
            Xt = X
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)
            return self.steps[-1][-1].predict(Xt, Y_possible)


class Proj(BaseEstimator, TransformerMixin):
    """Projection metaestimator for list-of-dict kind of data"""
    def __init__(self, transf, key):
        self.transf = transf
        self.key = key

    def fit(self, X, y=None):
        return self.transf.fit([x[self.key] for x in X], y)

    def transform(self, X):
        return self.transf.transform([x[self.key] for x in X])

    def fit_transform(self, X, y=None):
        return self.transf.fit_transform([x[self.key] for x in X], y)

    def get_feature_names(self):
        return self.transf.get_feature_names()


class LengthFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _doc_features(self, doc):
        n_sents = len(doc)
        all_toks = [tok for sent in doc for tok in sent]
        n_tokens = len(all_toks)
        n_types = len(set(all_toks))
        type_token_ratio = n_tokens / float(n_types)
        return np.array([n_sents, n_tokens, n_types, type_token_ratio],
                        dtype=np.float)

    def transform(self, X, y=None):
        #x in X is a list of sents
        return np.row_stack([self._doc_features(doc) for doc in X])

    def get_feature_names(self):
        return ['n_sents', 'n_tokens', 'n_types', 'type_token_ratio']


class NgramLolAnalyzer(object):
    def _word_ngrams(self, tokens, stop_words=None):
        """Turn tokens into a sequence of n-grams after stop words filtering"""
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]
        if self.lower:
            tokens = [w.lower() for w in tokens]
        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(" ".join(original_tokens[i: i + n]))
        return tokens

    def __init__(self, ngram_range, lower=False):
        self.ngram_range=ngram_range
        self.lower = lower

    def __call__(self, doc):
        return [feature for sentence in doc
                for feature in self._word_ngrams(sentence)]


with open(sys.argv[1]) as f:
    entries = json.load(f)

# some buggy docs are empty
entries = [entry for entry in entries if len(entry['lemmas'])]

X = np.array(entries)
Y = np.array([doc['interval'] for doc in entries])
Y_possible = np.array([doc['all_fine_intervals'] for doc in entries])

X, Y, Y_possible = shuffle(X, Y, Y_possible, random_state=0)

# make it run fast
X = X[:100]
Y = Y[:100]
Y_possible = Y_possible[:100]

pipe = MyPipeline([('vect', Proj(LengthFeatures(), key='lemmas')),
                   ('scale', StandardScaler(with_mean=False, with_std=True)),
                   ('clf', IntervalLogisticRegression(n_neighbors=10))])


grid = GridSearchCV(pipe,
                    dict(clf__C=np.logspace(-3, 3, 3)),
                    verbose=True, cv=KFold(len(X), n_folds=5),
                    scoring=semeval_interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible),
                    n_jobs=1)

grid.fit(X, Y)
print grid.best_score_, grid.best_params_
print grid.best_estimator_.steps[-1][-1].coef_.ravel()


pos_vect = TfidfVectorizer(use_idf=False, norm='l1', min_df=5, max_df=0.9,
                           analyzer=NgramLolAnalyzer(ngram_range=(2, 3),
                                                     lower=False))

union = FeatureUnion([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                      ('pos', Proj(pos_vect, key='pos'))])

pipe = MyPipeline([('union', union),
                   ('scale', StandardScaler(with_mean=False, with_std=True)),
                   ('clf', IntervalLogisticRegression(n_neighbors=10))])


grid = GridSearchCV(pipe,
                    dict(clf__C=np.logspace(-3, 3, 3)),
                    verbose=True, cv=KFold(len(X), n_folds=5),
                    scoring=semeval_interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible),
                    n_jobs=1)

grid.fit(X, Y)
print grid.best_score_, grid.best_params_

feature_names = grid.best_estimator_.steps[0][1].get_feature_names()
coef = grid.best_estimator_.steps[-1][-1].coef_.ravel()

for idx in np.argsort(-coef)[:10]:
    print("{:.2f}\t{}".format(coef[idx], feature_names[idx]))

