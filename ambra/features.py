import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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


class NgramLolAnalyzer(BaseEstimator):
    """Analyzer for pre-tokenized list-of-lists sentences-words structures"""
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
