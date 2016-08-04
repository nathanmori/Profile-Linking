# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

class dist_diff(object):

    def __init__(self, diffs):
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
