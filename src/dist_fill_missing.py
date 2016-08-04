# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

class dist_fill_missing(object):
    """"""

    def __init__(self, fill_with='mean'):
        """"""

        self.fill_with = fill_with

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
