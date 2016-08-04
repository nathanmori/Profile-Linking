# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pandas as pd
from sklearn.preprocessing import StandardScaler
import pdb


class scaler(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        self.ss = StandardScaler().fit(df_X_train.values)

        return self

    def transform(self, df_X, y=None):
        """"""

        scaled_arr = self.ss.transform(df_X.values)
        df_X = pd.DataFrame(scaled_arr, columns=df_X.columns)

        return df_X
