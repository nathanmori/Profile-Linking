# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdb


class cosine(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, (df_X_train, X_train_github, X_train_meetup), y=None):
        """"""

        return self

    def transform(self, (df_X, X_github, X_meetup), y=None):
        """"""

        df_X['text_sim'] = [cosine_similarity(x1, x2) for x1, x2 in
                                  zip(X_github_tfidf, X_meetup_tfidf)]

        return df_X
