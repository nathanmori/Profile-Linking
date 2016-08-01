import numpy as np
import pdb


class zero(object):
    """"""

    def __init__(self):
        """"""

        pass

    def fit(self, df_X_train, y=None):
        """"""

        print '\n\nfit: fill_missing_texts', df_X_train.head()

        """ NEED TO ADDRESS POSSIBILITY THAT iloc[0] is empty, figure a better
            way to get text vect length """
        self.text_vect_len = len(df_X_train.github_text.iloc[0].split())
        X_train_github = np.array([map(int, x.split()) if type(x) == str
                                                else [0] * self.text_vect_len
                             for x in df_X_train['github_text']])
        X_train_meetup = np.array([map(int, x.split()) if type(x) == str
                                                else [0] * self.text_vect_len
                             for x in df_X_train['meetup_text']])

        return self

    def transform(self, df_X, y=None):
        """"""

        print '\n\ntransform: fill_missing_texts', df_X.head()

        """ Convert text vectors from strings to list of ints,
            fill missing values with empty text
        CONSIDER leaving empty and fill missing cosine sims with mean cosine
            sim after calc'd """
        X_github = np.array([map(int, x.split()) if type(x) == str
                                                 else [0] * self.text_vect_len
                             for x in df_X['github_text']])
        X_meetup = np.array([map(int, x.split()) if type(x) == str
                                                 else [0] * self.text_vect_len
                             for x in df_X['meetup_text']])

        return df_X
