# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from name_tools import match
from copy import deepcopy
import pdb



class drop_github_meetup(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        return self

    def transform(self, df_X_input, y=None):
        """"""

        df_X = df_X_input.drop(['github', 'meetup'], axis=1, inplace=False)

        return df_X


class dist_fill_missing(object):
    """"""

    def __init__(self, fill_with='mean'):
        """"""

        self.fill_with = fill_with
        #validate

    def fit(self, df_X_train, y=None):
        """"""

        self.dist_cols = [col for col in df_X_train.columns if 'dist' in col]

        if self.fill_with == 'mean':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).mean()
        else:
            self.fill_vals = np.empty(len(self.dist_cols))
            self.fill_vals.fill(self.fill_with)

        return self

    def transform(self, df_X, y=None):
        """"""

        df_X[self.dist_cols] = df_X[self.dist_cols].fillna(self.fill_vals) \
                                        .apply(lambda ser: ser.apply(float))

        return df_X


class dist_diff(object):

    def __init__(self, diffs='all'):
        """"""

        self.diffs = diffs
        #validate

    def fit(self, df_X_train, y=None):
        """"""

        self.dist_cols = [col for col in df_X_train.columns if 'dist' in col]

        return self

    def transform(self, df_X, y=None):
        """"""

        if self.diffs == 'all':
            for i, col1 in enumerate(self.dist_cols, start=1):
                for col2 in self.dist_cols[i:]:
                    df_X['DIFF:' + col1 + '-' + col2] = df_X[col1] - df_X[col2]

        return df_X


class text_fill_missing(object):
    """"""

    def __init__(self, zero=True):
        """"""

        self.zero = zero
        ### MAY NOT NEED
        #validate

    def fit(self, df_X_train, y=None):
        """"""

        i = 0
        while type(df_X_train.github_text.iloc[i]) != str:
            i += 1
        self.text_vect_len = len(df_X_train.github_text.iloc[i].split())

        return self

    def transform(self, df_X_input):
        """"""

        df_X = df_X_input.copy()

        df_X['github_text_missing'] = df_X['github_text'].apply(
                                        lambda x: type(x) != str)
        df_X['meetup_text_missing'] = df_X['meetup_text'].apply(
                                        lambda x: type(x) != str)
        df_X['text_missing'] = df_X['github_text_missing'] | \
                               df_X['meetup_text_missing']

        fill_val = [0] * self.text_vect_len# if self.zero else np.nan
        X_github = np.array([map(int, x.split())
                                if type(x) == str
                                else fill_val
                                for x
                                in df_X['github_text']])
        X_meetup = np.array([map(int, x.split())
                                if type(x) == str
                                else fill_val
                                for x
                                in df_X['meetup_text']])

        df_X.drop(['github_text', 'meetup_text'], axis=1, inplace=True)

        return (df_X, X_github, X_meetup)


class text_idf(object):
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

        if self.idf == 'no':
            return (df_X, csr_matrix(X_github), csr_matrix(X_meetup))

        X_github_tfidf = self.tfidf_github.transform(X_github)
        X_meetup_tfidf = self.tfidf_meetup.transform(X_meetup)

        if self.idf == 'yes':
            return (df_X, X_github_tfidf, X_meetup_tfidf)

        return (df_X, csr_matrix(X_github), csr_matrix(X_meetup),
                X_github_tfidf, X_meetup_tfidf)


class text_aggregate(object):
    """"""

    def __init__(self, refill_missing=False, drop_missing_bools=False):
        """"""

        #validate
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


class name_similarity(object):
    """"""

    def __init__(self):
        """"""

        """ ADD option for name_tools, first, last, full COMBOS ? """
        pass

    def fit(self, df_X_train, y=None):
        """"""

        return self

    def transform(self, df_X_input, y=None):
        """"""

        df_X = df_X_input.copy()

        df_X['name_sim'] = df_X.apply(lambda row: match(row['github_name'],
                                                        row['meetup_name']
                                                       ),
                                      axis=1
                                     )
        df_X.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

        return df_X


class scaler(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        self.ss = StandardScaler().fit(df_X_train.values)

        return self

    def transform(self, df_X, y=None):
        """"""

        scaled_arr = self.ss.transform(df_X.values)
        df_X = pd.DataFrame(scaled_arr, columns=df_X.columns)

        return df_X


class df_to_array(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        return self

    def transform(self, df_X, y=None):
        """"""

        X = df_X.values
        return X


class UD_pipe(object):
    """"""

    def __init__(self,
                 mod,
                 dist_fill_with='mean',
                 dist_diffs='all',
                 text_fill_zero=True,
                 idf='both',
                 text_refill_missing=False,
                 text_drop_missing_bools=False
                 ):
        """"""
        self.params = {'dist_fill_with': dist_fill_with,
                       'dist_diffs': dist_diffs,
                       'text_fill_zero': text_fill_zero,
                       'idf': idf,
                       'text_refill_missing': text_refill_missing,
                       'text_drop_missing_bools': text_drop_missing_bools,
                       'mod': mod
        }

        self.pipe = Pipeline([('drop_github_meetup',
                                drop_github_meetup()),
                              ('dist_fill_missing',
                                dist_fill_missing(
                                    self.params['dist_fill_with'])),
                              ('dist_diff',
                                dist_diff(
                                    self.params['dist_diffs'])),
                              ('text_fill_missing',
                                text_fill_missing(
                                    self.params['text_fill_zero'])),
                              ('text_idf',
                                text_idf(
                                    self.params['idf'])),
                              ('text_aggregate',
                                text_aggregate(
                                    self.params['text_refill_missing'],
                                    self.params['text_drop_missing_bools'])),
                              ('name_similarity',
                                name_similarity()),
                              ('scaler',
                                scaler()),
                              ('df_to_array',
                                df_to_array()),
                              ('mod',
                                mod)
                             ]
                            )


    def get_params(self, deep=True):
        """"""

        if deep:

            return deepcopy(self.params)

        return self.params


    def fit(self, df_X, y=None):
        """"""

        self.pipe.fit(df_X, y)
        self.classes_ = self.pipe.named_steps['mod'].classes_

        return self


    def transform(self, df_X, y=None):
        """"""

        return self.pipe.transform(df_X)

    def predict_proba(self, df_X):
        """"""

        return self.pipe.predict_proba(df_X)
