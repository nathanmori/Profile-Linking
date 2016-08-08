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
from time import time
import pdb


def validate(key, val, alts):
    """"""

    if val not in alts:
        raise ValueError("%s=%s must be in %s" % (key, val, alts))
    else:
        print 'REPORT   validate: %s=%s OK' % (key, val)


class UD_transform_class(object):
    """"""

    def __init__(self):
        """"""

        self.params = {}

    def fit(self, *args):
        """"""

        return self

    def get_params(self, deep=True):
        """"""

        if deep:
            return deepcopy(self.params)

        return self.params

    def set_params(self, **params):
        """"""

        for key, value in params.iteritems():
            validate(key, value, self.valids[key])
            self.params[key] = value

    def fill_params(self):
        """"""

        for param in self.defaults:
            if param not in self.params:
                self.params[param] = self.defaults[param]


class drop_github_meetup(UD_transform_class):
    """"""

    def transform(self, df_X_input):
        """"""

        df_X = df_X_input.drop(['github', 'meetup'], axis=1, inplace=False)

        return df_X


class dist_fill_missing(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'fill_with': 'median'}
        self.valids = {'fill_with': ['mean', 'median', 'min', 'max']}

    def fit(self, df_X_train, y=None):
        """"""

        self.fill_params()

        self.dist_cols = [col for col in df_X_train.columns if 'dist' in col]

        if self.params['fill_with'] == 'mean':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).mean()

        elif self.params['fill_with'] == 'min':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).min()

        elif self.params['fill_with'] == 'max':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).max()

        elif self.params['fill_with'] == 'median':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                         lambda ser: ser.apply(float)).median()

        return self

    def transform(self, df_X):
        """"""

        df_X[self.dist_cols] = df_X[self.dist_cols].fillna(self.fill_vals) \
                                        .apply(lambda ser: ser.apply(float))

        return df_X


class dist_diff(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'include': 'all'}
        self.valids = {'include': ['all', 'none', 'ignore_min']}

    def fit(self, df_X_train, y=None):
        """"""

        self.fill_params()

        if self.params['include'] == 'ignore_min':
            self.dist_cols = [col for col in df_X_train.columns
                                  if ('dist' in col and
                                      'min' not in col)]
        else:
            self.dist_cols = [col for col in df_X_train.columns
                                  if 'dist' in col]

        return self

    def transform(self, df_X):
        """"""

        if self.params['include'] != 'none':
            for i, col1 in enumerate(self.dist_cols, start=1):
                for col2 in self.dist_cols[i:]:
                    df_X['DIFF:' + col1 + '-' + col2] = df_X[col1] - df_X[col2]

        return df_X


class text_fill_missing(UD_transform_class):
    """"""

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

        fill_val = [0] * self.text_vect_len
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



class text_idf(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'idf': 'yes'}
        self.valids = {'idf': ['yes', 'no', 'both']}

    def fit(self, (df_X_train, X_train_github, X_train_meetup), y=None):
        """"""

        self.fill_params()

        if self.params['idf'] in ['yes', 'both']:
            self.tfidf_github = TfidfTransformer().fit(X_train_github)
            self.tfidf_meetup = TfidfTransformer().fit(X_train_meetup)

        return self

    def transform(self, (df_X, X_github, X_meetup)):
        """"""

        if self.params['idf'] == 'no':
            return (df_X, csr_matrix(X_github), csr_matrix(X_meetup), None,
                                                                      None)

        X_github_tfidf = self.tfidf_github.transform(X_github)
        X_meetup_tfidf = self.tfidf_meetup.transform(X_meetup)

        if self.params['idf'] == 'yes':
            return (df_X, None, None, X_github_tfidf, X_meetup_tfidf)

        return (df_X, csr_matrix(X_github), csr_matrix(X_meetup),
                X_github_tfidf, X_meetup_tfidf)


class text_aggregate(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'refill_missing': False,
                         'cosine_only': True,
                         'drop_missing_bools': True}
        self.valids = {'refill_missing': [True, False],
                       'cosine_only': [True, False],
                       'drop_missing_bools': [True, False]}

    def fit(self, (df_X_input, X_github, X_meetup,
                   X_github_tfidf, X_meetup_tfidf), y=None):
        """"""

        self.fill_params()

        if self.params['refill_missing']:

            df_X = df_X_input.copy()

            if X_github is not None:

                df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                      in zip(X_github, X_meetup)]

                if not self.params['cosine_only']:
                    df_X['text_norm_github'] = [norm(x) for x in X_github]
                    df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
                    df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                             df_X['text_norm_meetup']
                    df_X['text_dot'] = [x1.dot(x2.T)[0,0]
                                        for x1, x2
                                        in zip(X_github, X_meetup)]

            if X_github_tfidf is not None:

                df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0]
                                          for x1, x2
                                          in zip(X_github_tfidf,
                                                 X_meetup_tfidf)]

                if not self.params['cosine_only']:
                    df_X['text_norm_github_tfidf'] = [norm(x) for x
                                                      in X_github_tfidf]
                    df_X['text_norm_meetup_tfidf'] = [norm(x) for x
                                                      in X_meetup_tfidf]
                    df_X['DIFF:text_norm_tfidf'] = \
                        df_X['text_norm_github_tfidf'] - \
                        df_X['text_norm_meetup_tfidf']
                    df_X['text_dot_tfidf'] = [x1.dot(x2.T)[0,0]
                                              for x1, x2
                                              in zip(X_github_tfidf,
                                                     X_meetup_tfidf)]

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
                if col in df_X.columns:
                    self.refill_means[col] = df_X[col] \
                                            [~ df_X['text_missing']].mean()

        return self

    def transform(self, (df_X_input, X_github, X_meetup, X_github_tfidf,
                         X_meetup_tfidf)):
        """"""

        df_X = df_X_input.copy()

        if X_github is not None:

            df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                in zip(X_github, X_meetup)]

            if not self.params['cosine_only']:
                df_X['text_norm_github'] = [norm(x) for x in X_github]
                df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
                df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                         df_X['text_norm_meetup']
                df_X['text_dot'] = [x1.dot(x2.T)[0,0] for x1, x2
                                    in zip(X_github, X_meetup)]

        if X_github_tfidf is not None:

            df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                      in zip(X_github_tfidf, X_meetup_tfidf)]

            if not self.params['cosine_only']:
                df_X['text_norm_github_tfidf'] = [norm(x) for x
                                                  in X_github_tfidf]
                df_X['text_norm_meetup_tfidf'] = [norm(x) for x
                                                  in X_meetup_tfidf]
                df_X['DIFF:text_norm_tfidf'] = \
                    df_X['text_norm_github_tfidf'] - \
                    df_X['text_norm_meetup_tfidf']
                df_X['text_dot_tfidf'] = [x1.dot(x2.T)[0,0] for x1, x2
                                          in zip(X_github_tfidf,
                                                 X_meetup_tfidf)]

        if self.params['refill_missing']:
            for col in self.refill_cols:
                if col in df_X.columns:
                    df_X[col][df_X['text_missing']] = self.refill_means[col]

        missing_bool_cols = ['github_text_missing',
                             'meetup_text_missing',
                             'text_missing']
        if self.params['drop_missing_bools']:
            df_X.drop(missing_bool_cols, axis=1, inplace=True)
        else:
            for col in missing_bool_cols:
                df_X[col] = df_X[col].apply(int)

        return df_X


class name_similarity(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'fullname': False,
                         'firstname': False,
                         'lastname': True,
                         'calc': False}
        self.valids = {'fullname': [True, False],
                       'firstname': [True, False],
                       'lastname': [True, False],
                       'calc': [True, False]}

    def fit(self, df_X, y=None):
        """"""

        self.fill_params()

        return self

    def transform(self, df_X_input):
        """"""

        df_X = df_X_input.copy()

        if not self.params['fullname']:
            df_X.drop(['fullname_similarity'], axis=1, inplace=True)

        if not self.params['firstname']:
            df_X.drop(['firstname_similarity'], axis=1, inplace=True)

        if not self.params['lastname']:
            df_X.drop(['lastname_similarity'], axis=1, inplace=True)

        if self.params['calc']:
            df_X['name_sim'] = df_X.apply(lambda row: match(row['github_name'],
                                                            row['meetup_name']
                                                           ),
                                          axis=1)

        df_X.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

        return df_X


class scaler(UD_transform_class):
    """"""

    def fit(self, df_X_train, y=None):
        """"""

        self.ss = StandardScaler().fit(df_X_train.values)

        return self

    def transform(self, df_X):
        """"""

        scaled_arr = self.ss.transform(df_X.values)
        df_X = pd.DataFrame(scaled_arr, columns=df_X.columns)

        return df_X


class drop_feat(UD_transform_class):
    """"""

    def __init__(self):
        """"""

        self.params = {}
        self.defaults = {'num': None}
        self.valids = {'num': [None] + range(50)}

    def fit(self, df_X, y=None):
        """"""

        self.fill_params()

        return self

    def transform(self, df_X):
        """"""

        if (self.params['num'] is not None) and \
           (self.params['num'] < df_X.shape[1]):

            col = df_X.columns[self.params['num']]

            df_X.drop([col], axis=1, inplace=True)

        return df_X


class df_to_array(UD_transform_class):
    """"""

    def fit(self, df_X, y=None):
        """"""

        self.feats = df_X.columns

        return self

    def transform(self, df_X):
        """"""

        X = df_X.values
        return X
