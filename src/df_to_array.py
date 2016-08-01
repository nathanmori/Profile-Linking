class df_to_array(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        print '\n\nfit: df_to_array', df_X_train.head()

        return self

    def transform(self, df_X, y=None):
        """"""

        print '\n\ntransform: df_to_array', df_X.head()

        X = df_X.values
        return X
