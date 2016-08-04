# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from sklearn.pipeline import Pipeline
import text_fill_missing
import text_idf
import text_aggregate

""" DELETE FILE """


class text(object):
    """"""

    def __init__(self):
        """"""

        self.pipe = Pipeline([('fill_missing',
                                    text_fill_missing.fill()),
                              ('idf', text_idf.idf(idf='both')),
                              ('aggregate',
                                text_aggregate.all(refill_missing=True))])

    def fit(self, df_X_train, y_train=None):
        """"""

        df_X_train_fit = df_X_train.copy()
        self.pipe.fit(df_X_train_fit, y_train)

        return self

    def transform(self, df_X):
        """"""

        df_transform = self.pipe.transform(df_X)

        return df_transform
