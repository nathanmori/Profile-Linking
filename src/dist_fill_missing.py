# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

class mean(object):
    """"""

    def __init__(self):
        """"""

        self.dist_cols = ['min_dist_km', 'avg_dist_km', 'median_dist_km', 'max_dist_km']

    def fit(self, df_X_train, y=None):
        """"""

        self.dist_means = df_X_train[self.dist_cols].dropna().apply(lambda ser: ser.apply(float)).mean()

        return self

    def transform(self, df_X, y=None):
        """"""

        df_X[self.dist_cols] = df_X[self.dist_cols].fillna(self.dist_means) \
                                        .apply(lambda ser: ser.apply(float))

        return df_X