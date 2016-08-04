# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from name_tools import match


class name_similarity(object):
    """"""

    def __init__(self):
        """"""

        """ ADD option for name_tools, first, last, full COMBOS ? """
        pass

    def fit(self, df_X_train, y=None):
        """"""

        return self

    def transform(self, df_X_input, y=None):
        """"""

        df_X = df_X_input.copy()

        df_X['name_sim'] = df_X.apply(lambda row: match(row['github_name'],
                                                        row['meetup_name']
                                                       ),
                                      axis=1
                                     )
        df_X.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

        return df_X
