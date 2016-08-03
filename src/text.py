# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from sklearn.pipeline import Pipeline
import text_fill_missing
import text_idf
import text_aggreagate


class all(object):
    """"""

    def __init__(self):
        """"""

        self.pipe = Pipeline([('fill_missing', text_fill_missing.zero(),
                                #need a both 
                              ('idf', text_idf.class()),
                              ('aggregate', text_aggregate())])

    def fit(self, df_X_train, y_train=None):
        """"""

        self.pipe.fit(df_X_train, y_train)

        return self

    def transform(self, df_X, y=None):
        """"""

        df_transform = self.pipe.transform(df_X, y):

        return df_transform
