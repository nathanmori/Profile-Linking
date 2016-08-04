# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>


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
