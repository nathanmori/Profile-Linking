# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
import pdb


class zero(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        """ NEED TO ADDRESS POSSIBILITY THAT iloc[0] is empty, figure a better
            way to get text vect length """
        self.text_vect_len = len(df_X_train.github_text.iloc[0].split())
        X_train_github = np.array([map(int, x.split()) if type(x) == str
                                                else [0] * self.text_vect_len
                             for x in df_X_train['github_text']])
        X_train_meetup = np.array([map(int, x.split()) if type(x) == str
                                                else [0] * self.text_vect_len
                             for x in df_X_train['meetup_text']])

        return self

    def transform(self, df_X):
        """"""

        """ Convert text vectors from strings to list of ints,
            fill missing values with empty text
        CONSIDER leaving empty and fill missing cosine sims with mean cosine
            sim after calc'd """

        X_github = np.array([map(int, x.split()) if type(x) == str
                                                 else [0] * self.text_vect_len
                             for x in df_X['github_text']])
        X_meetup = np.array([map(int, x.split()) if type(x) == str
                                                 else [0] * self.text_vect_len
                             for x in df_X['meetup_text']])
        df_X.drop(['github_text', 'meetup_text'], axis=1, inplace=True)

        """NOTE UNIQUE INTERFACE """

        return (df_X, X_github, X_meetup)
