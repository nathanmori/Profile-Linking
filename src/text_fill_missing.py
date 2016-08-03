# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
import pdb


def check_nans(arr):
    print '%d missing values' % np.isnan(arr).sum()


class fill(object):
    """"""

    def __init__(self, zero=True):
        """"""

        self.zero = zero

    def fit(self, df_X_train, y=None):
        """"""

        print 'fit missing'

        i = 0
        while type(df_X_train.github_text.iloc[i]) != str:
            i += 1
        self.text_vect_len = len(df_X_train.github_text.iloc[i].split())

        return self

    def transform(self, df_X):
        """"""

        df_X['github_text_missing'] = df_X['github_text'].apply(
                                        lambda x: type(x) != str)
        df_X['meetup_text_missing'] = df_X['meetup_text'].apply(
                                        lambda x: type(x) != str)
        df_X['text_missing'] = df_X['github_text_missing'] | \
                               df_X['meetup_text_missing']


        """ FIGURE OUT WHY NONE MISSING ON SECOND TIME """

        print 'trans missing'
        print df_X['github_text_missing'].value_counts()
        print df_X['meetup_text_missing'].value_counts()
        print df_X['text_missing'].value_counts()

        fill_val = [0] * self.text_vect_len if self.zero else np.nan
            
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
