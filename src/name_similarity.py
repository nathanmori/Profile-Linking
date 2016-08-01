from name_tools import match


class name_tools_match(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        print '\n\nfit: name_similarity', df_X_train.head()

        return self

    def transform(self, df_X, y=None):
        """"""

        print '\n\ntransform: name_similarity', df_X.head()

        df_X['name_sim'] = df_X.apply(lambda row: match(row['github_name'],
                                                        row['meetup_name']),
                                      axis=1)
        df_X.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

        return df_X
