# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

class all(object):

    def __init__(self):
        """"""
        
        pass

    def fit(self, df_X_train, y=None):
        """"""

        self.dist_cols = [col for col in df_X_train.columns if 'dist' in col]

        return self

    def transform(self, df_X, y=None):
        """"""

        for i, col1 in enumerate(self.dist_cols):
            for j, col2 in enumerate(self.dist_cols[1:], start=1):
                df_X[col1 + '-' + col2] = df_X[col1] - df_X[col2]

        return df_X
