from __future__ import print_function
import sys
import json
import numpy as np

from scipy.stats import sem

from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2

from ambra.classifiers import IntervalLogisticRegression
from ambra.interval_scoring import semeval_interval_scorer
from ambra.temporal_feature_extraction import get_temporal_feature

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
        self.transf = self.transf.fit([x[self.key] for x in X], y)
        return self

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
        #type_token_ratio = n_tokens / float(n_types)
        return np.array([n_sents, n_tokens, n_types,
                         #type_token_ratio
                         ],
                        dtype=np.float)

    def transform(self, X, y=None):
        #x in X is a list of sents
        return np.row_stack([self._doc_features(doc) for doc in X])

    def get_feature_names(self):
        return ['n_sents', 'n_tokens', 'n_types',
        #'type_token_ratio'
        ]


class StylisticFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, lower=True):
        self.lower = lower

    def fit(self, X, y=None):
        return self

    def _doc_features(self, doc):
        # doc is a dict
        tokens = doc['tokens']
        lemmas = doc['lemmas']
        all_tokens = [w.lower() if self.lower else w
                      for sent in tokens for w in sent]
        all_lemmas = [w.lower() if self.lower else w
                      for sent in lemmas for w in sent]
        avg_sent_len = np.mean([len(sent) for sent in tokens])
        avg_word_len = np.mean([len(w) for w in tokens])
        lex_density = len(set(all_tokens)) / len(all_tokens)
        lex_richness = len(set(all_lemmas)) / len(all_lemmas)
        return np.array([avg_sent_len, avg_word_len, lex_density, lex_richness],
                        dtype=np.float)

    def transform(self, X, y=None):
        #x in X is a list of sents
        return np.row_stack([self._doc_features(doc) for doc in X])

    def get_feature_names(self):
        return ['ASL', 'AWL', 'LD', 'LR']

class TemporalFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #x in X is a list of sents
        return np.row_stack([get_temporal_feature(doc) for doc in X])

    def get_feature_names(self):
        return ['YEAR']

class NgramLolAnalyzer(BaseEstimator):
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

    def __init__(self, ngram_range=(1,1), lower=False):
        self.ngram_range=ngram_range
        self.lower = lower

    def __call__(self, doc):
        return [feature for sentence in doc
                for feature in self._word_ngrams(sentence)]


class MySelectKBest(SelectKBest):
    def fit(self, X, y):
	y_flat = [get_coarse_interval(np.mean(yelem)) for yelem in y]
        return super(MySelectKBest, self).fit(X, y_flat)

def get_coarse_interval(year):
    year = int(year)
    lower = year / 100 * 100 + 50 * (year % 100 >= 50)
    return str(lower) + "-" + str(lower + 50)

with open(sys.argv[1]) as f:
    entries = json.load(f)

# some buggy docs are empty
entries = [entry for entry in entries if len(entry['lemmas'])]

X = np.array(entries)
Y = np.array([doc['interval'] for doc in entries])
Y_possible = np.array([doc['all_fine_intervals'] for doc in entries])

X, Y, Y_possible = shuffle(X, Y, Y_possible, random_state=0)

# make it run fast
limit = None  # 500
limit_pairs = 0.01
if limit:
    X = X[:limit]
    Y = Y[:limit]
    Y_possible = Y_possible[:limit]

print("Temporal feature")
print("===============")
pipe = MyPipeline([('temp', TemporalFeature()),
                   ('clf', IntervalLogisticRegression(n_neighbors=10,
                                                      limit_pairs=limit_pairs,
                                                      random_state=0))])

grid = GridSearchCV(pipe,
                    dict(clf__C=np.logspace(-3, 3, 7)),
                    verbose=False, cv=KFold(len(X), n_folds=5),
                    scoring=semeval_interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible),
                    n_jobs=1)

grid.fit(X, Y)
grid_scores = [k.mean_validation_score for k in grid.grid_scores_]
print("{:.3f} +/- {:.4f}".format(grid.best_score_, sem(grid_scores)))
print(grid.best_estimator_.steps[-1][-1].n_pairs_, " total pairs.")
print(grid.best_estimator_.steps[-1][-1].coef_.ravel())

print("Length features")
print("===============")
pipe = MyPipeline([('vect', Proj(LengthFeatures(), key='lemmas')),
                   ('scale', StandardScaler(with_mean=False, with_std=True)),
                   ('clf', IntervalLogisticRegression(n_neighbors=10,
                                                      limit_pairs=limit_pairs,
                                                      random_state=0))])

grid = GridSearchCV(pipe,
                    dict(clf__C=np.logspace(-3, 3, 7)),
                    verbose=False, cv=KFold(len(X), n_folds=5),
                    scoring=semeval_interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible),
                    n_jobs=1)

grid.fit(X, Y)
grid_scores = [k.mean_validation_score for k in grid.grid_scores_]
print("{:.3f} +/- {:.4f}".format(grid.best_score_, sem(grid_scores)))
print(grid.best_estimator_.steps[-1][-1].n_pairs_, " total pairs.")
print(grid.best_estimator_.steps[-1][-1].coef_.ravel())

print()
print("Stylistic features")
print("==================")

union = FeatureUnion([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                      ('style', StylisticFeatures())])

pipe = MyPipeline([('vect', union),
                   ('scale', StandardScaler(with_mean=False, with_std=True)),
                   ('clf', IntervalLogisticRegression(n_neighbors=10,
                                                      limit_pairs=limit_pairs,
                                                      random_state=0))])

grid = GridSearchCV(pipe,
                    dict(clf__C=np.logspace(-3, 3, 7)),
                    verbose=False, cv=KFold(len(X), n_folds=5),
                    scoring=semeval_interval_scorer,
                    scorer_params=dict(Y_possible=Y_possible),
                    n_jobs=1)

grid.fit(X, Y)
grid_scores = [k.mean_validation_score for k in grid.grid_scores_]
print("{:.3f} +/- {:.4f}".format(grid.best_score_, sem(grid_scores)))
print(grid.best_estimator_.steps[-1][-1].n_pairs_, " total pairs.")
print(grid.best_estimator_.steps[-1][-1].coef_.ravel())

print()
print("Lex and POS features")
print("====================")

vectorizer = TfidfVectorizer(use_idf=False, norm='l1', analyzer=NgramLolAnalyzer(lower=False))

union = FeatureUnion([('lenghts', Proj(LengthFeatures(), key='lemmas')),
                      ('style', StylisticFeatures()),
                      ('pos', Proj(vectorizer, key='pos')),
                      ('tokens', Proj(vectorizer, key='tokens'))])

pipe = MyPipeline([('union', union),
                   ('scale', StandardScaler(with_mean=False, with_std=True)),
                   ('fs', MySelectKBest(chi2)),
                   ('clf', IntervalLogisticRegression(n_neighbors=10,
                                                      limit_pairs=limit_pairs,
                                                      random_state=0))])


grid_params = {
	'clf__C' : np.logspace(-3, 3, 7),
	'union__pos__transf__min_df' : [10, 5, 1],
	'union__pos__transf__max_df' : [1.0, 0.99, 0.9, 0.8],
    'union__pos__transf__analyzer__ngram_range' : [(2, 3), (2, 2)],
	'union__tokens__transf__min_df' : [10, 5, 1],
	'union__tokens__transf__max_df' : [1.0, 0.99, 0.9, 0.8],
    'union__tokens__transf__analyzer__ngram_range' : [(1, 3), (1, 2), (1, 1)],
    'fs__k' : [50, 200, 400]
}

grid = RandomizedSearchCV(pipe,
                          grid_params,
                          verbose=True, cv=KFold(len(X), n_folds=5),
                          scoring=semeval_interval_scorer,
                          scorer_params=dict(Y_possible=Y_possible),
                          n_iter=4,
                          n_jobs=1,
                          random_state=0)

grid.fit(X, Y)
grid_scores = [k.mean_validation_score for k in grid.grid_scores_]
print("{:.3f} +/- {:.4f}".format(grid.best_score_, sem(grid_scores)))
print(grid.best_params_)

feature_names = grid.best_estimator_.steps[0][1].get_feature_names()
feature_names = np.array(feature_names)[
    grid.best_estimator_.steps[2][1].get_support()]
coef = grid.best_estimator_.steps[-1][-1].coef_.ravel()

for idx in np.argsort(-np.abs(coef))[:50]:
    print("{:.2f}\t{}".format(coef[idx], feature_names[idx]))
