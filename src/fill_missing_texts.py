import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
        self.tfidf_github = TfidfTransformer().fit(X_train_github)
        self.tfidf_meetup = TfidfTransformer().fit(X_train_meetup)

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
        X_github_tfidf = self.tfidf_github.transform(X_github)
        X_meetup_tfidf = self.tfidf_meetup.transform(X_meetup)
        df_X['text_sim'] = [cosine_similarity(x1, x2) for x1, x2 in
                                  zip(X_github_tfidf, X_meetup_tfidf)]
        df_X.drop(['github_text', 'meetup_text'], axis=1, inplace=True)

        return df_X
