# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdb
from scipy.sparse import csr_matrix


class idf(object):
    """"""

    def __init__(self, idf='both'):
        """"""

        if idf not in ['yes', 'no', 'both']:
            raise ValueError("idf must be in ['yes', 'no', 'both']")

        self.idf = idf

    def fit(self, (df_X_train, X_train_github, X_train_meetup), y=None):
        """"""
        if self.idf in ['yes', 'both']:
            self.tfidf_github = TfidfTransformer().fit(X_train_github)
            self.tfidf_meetup = TfidfTransformer().fit(X_train_meetup)

        return self

    def transform(self, (df_X, X_github, X_meetup)):
        """"""

        pdb.set_trace()

        if self.idf == 'no':
            return (df_X, csr_matrix(X_github), csr_matrix(X_meetup))

        X_github_tfidf = self.tfidf_github.transform(X_github)
        X_meetup_tfidf = self.tfidf_meetup.transform(X_meetup)

        if self.idf == 'yes':
            return (df_X, X_github_tfidf, X_meetup_tfidf)

        return (df_X, csr_matrix(X_github), csr_matrix(X_meetup),
                X_github_tfidf, X_meetup_tfidf)
