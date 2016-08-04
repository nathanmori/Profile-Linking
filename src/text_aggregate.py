# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
from scipy.sparse.linalg import norm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdb


class text_aggregate(object):
    """"""

    def __init__(self, refill_missing=False, drop_missing_bools=False):
        """"""

        self.refill_missing = refill_missing
        self.drop_missing_bools = drop_missing_bools

    def fit(self, (df_X_input, X_github, X_meetup,
                   X_github_tfidf, X_meetup_tfidf), y=None):
        """"""

        df_X = df_X_input.copy()

        df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2 in
                                  zip(X_github, X_meetup)]
        df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0] for x1, x2 in
                                  zip(X_github_tfidf, X_meetup_tfidf)]

        df_X['text_norm_github'] = [norm(x) for x in X_github]
        df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
        df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                 df_X['text_norm_meetup']

        df_X['text_norm_github_tfidf'] = [norm(x) for x in X_github_tfidf]
        df_X['text_norm_meetup_tfidf'] = [norm(x) for x in X_meetup_tfidf]
        df_X['DIFF:text_norm_tfidf'] = df_X['text_norm_github_tfidf'] - \
                                       df_X['text_norm_meetup_tfidf']

        df_X['text_dot'] = [np.dot(x1, x2.T)[0][0,0] for x1, x2
                                                    in zip(X_github, X_meetup)]
        df_X['text_dot_tfidf'] = [np.dot(x1, x2.T)[0][0,0] for x1, x2
                                        in zip(X_github_tfidf, X_meetup_tfidf)]


        if self.refill_missing:
            self.refill_cols = ['text_sim',
                                'text_sim_tfidf',
                                'text_norm_github',
                                'text_norm_meetup',
                                'DIFF:text_norm',
                                'text_norm_github_tfidf',
                                'text_norm_meetup_tfidf',
                                'DIFF:text_norm_tfidf',
                                'text_dot',
                                'text_dot_tfidf']
            self.refill_means = {}

            for col in self.refill_cols:
                self.refill_means[col] = df_X[col] \
                                            [~ df_X['text_missing']].mean()

            pass

        return self

    def transform(self, (df_X_input, X_github, X_meetup, X_github_tfidf,
                         X_meetup_tfidf)):
        """"""

        df_X = df_X_input.copy()

        df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2 in
                                  zip(X_github, X_meetup)]
        df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0] for x1, x2 in
                                  zip(X_github_tfidf, X_meetup_tfidf)]

        df_X['text_norm_github'] = [norm(x) for x in X_github]
        df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
        df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                 df_X['text_norm_meetup']

        df_X['text_norm_github_tfidf'] = [norm(x) for x in X_github_tfidf]
        df_X['text_norm_meetup_tfidf'] = [norm(x) for x in X_meetup_tfidf]
        df_X['DIFF:text_norm_tfidf'] = df_X['text_norm_github_tfidf'] - \
                                       df_X['text_norm_meetup_tfidf']

        df_X['text_dot'] = [np.dot(x1, x2.T)[0][0,0] for x1, x2
                                                    in zip(X_github, X_meetup)]
        df_X['text_dot_tfidf'] = [np.dot(x1, x2.T)[0][0,0] for x1, x2
                                        in zip(X_github_tfidf, X_meetup_tfidf)]

        if self.refill_missing:
            for col in self.refill_cols:
                df_X[col][df_X['text_missing']] = self.refill_means[col]

        missing_bool_cols = ['github_text_missing',
                             'meetup_text_missing',
                             'text_missing']
        if self.drop_missing_bools:
            df_X.drop(missing_bool_cols, axis=1, inplace=True)
        else:
            for col in missing_bool_cols:
                df_X[col] = df_X[col].apply(int)

        return df_X
