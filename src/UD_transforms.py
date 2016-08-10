# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from name_tools import match
from time import time
import pdb


def validate(key, val, valids):
    """
    Validate arguments.

    Check against list of valids for the parameter.

    Parameters
    ----------
    key : string
        Parameter being validated.

    val : any
        Argument passed in for parameter key.

    Returns
    -------
    None
    """

    if val not in valids:
        raise ValueError("%s=%s must be in %s" % (key, val, alts))


class UD_transform_class(object):
    """
    User-defined scikit-learn style fit/transform class.

    Skeleton parent class for all transforms used, including filling missing
    values and feature engineering.

    Parameters
    ----------
    None
    """

    def __init__(self):

        self.params = {}

    def fit(self, *args):
        """
        Empty scikit-learn style fit method for child classes requiring no fit
        operations. Used for Pipeline compatibility.

        Parameters
        ----------
        *args : args
            Additional parameters passed to the fit function of the estimator.

        Returns
        -------
        self : object
            Returns self.
        """

        return self

    def get_params(self, deep=True):
        """
        Get parameters of the estimator.

        Parameters
        ----------
        deep : bool, optional
            For compatibility with scikit-learn classes.

        Returns
        -------
        self.params : mapping of string to any
            Parameter names mapped to their values.
        """

        return self.params

    def set_params(self, **params):
        """
        Validate and set the parameters of the estimator.

        Parameters
        ----------
        **params : kwargs
            Parameter names mapped to the values to be set.

        Returns
        -------
        None
        """

        for key, value in params.iteritems():
            validate(key, value, self.valids[key])
            self.params[key] = value


class drop_github_meetup(UD_transform_class):
    """
    Drops github and meetup id #s from data.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    None
    """

    def transform(self, df_X):
        """
        Drop github and meetup ids from data.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X_new : pandas.DataFrame
            Transformed data.
        """

        df_X_new = df_X.drop(['github', 'meetup'], axis=1, inplace=False)

        return df_X_new


class dist_fill_missing(UD_transform_class):
    """
    Fills missing distance values.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    fill_with : str, default='median'
        Aggregation to be used to fill missing values.
    """

    def __init__(self, fill_with='median'):

        self.params = {'fill_with': fill_with}
        self.valids = {'fill_with': ['mean', 'median', 'min', 'max']}

    def fit(self, df_X_train, y=None):
        """
        Compute aggregations to be used to fill missing distances.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        self.dist_cols = [col for col in df_X_train.columns if 'dist' in col]

        if self.params['fill_with'] == 'mean':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).mean()

        elif self.params['fill_with'] == 'min':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).min()

        elif self.params['fill_with'] == 'max':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                           lambda ser: ser.apply(float)).max()

        elif self.params['fill_with'] == 'median':
            self.fill_vals = df_X_train[self.dist_cols].dropna().apply(
                                         lambda ser: ser.apply(float)).median()

        return self

    def transform(self, df_X):
        """
        Fill missing distance values.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.
        """

        df_X[self.dist_cols] = df_X[self.dist_cols].fillna(self.fill_vals) \
                                        .apply(lambda ser: ser.apply(float))

        return df_X


class dist_diff(UD_transform_class):
    """
    Calculates difference between max and min distances, and removes unkept
    distance columns.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    diffs : str
        Includes difference between max and min distances if 'range'.

    keep : str
        Distance column to use as a feature.
    """

    def __init__(self, diffs='range', keep='median'):

        self.params = {'diffs': diffs,
                       'keep': keep}
        self.valids = {'diffs': ['none', 'range'],
                       'keep': ['min', 'avg', 'median', 'max']}

    def fit(self, df_X_train, y=None):
        """
        Store distance columns to drop.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.

        """

        self.drop_cols = [col for col in df_X_train.columns
                              if ('dist' in col and
                                  self.params['keep'] not in col)]

        return self

    def transform(self, df_X):
        """
        Calculate difference between max and min distances, and removes unkept
        distance columns.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.

        """

        if self.params['diffs'] == 'range':
            df_X['dist_range_km'] = df_X['max_dist_km'] - df_X['min_dist_km']

        df_X.drop(self.drop_cols, axis=1, inplace=True)

        return df_X


class text_fill_missing(UD_transform_class):
    """
    Fills missing text values. Adds columns to track missing values.

    User-defined scikit-learn style fit/transform class.
    Parameters
    ----------

    """

    def fit(self, df_X_train, y=None):
        """
        Store length of text vectors.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        i = 0
        while type(df_X_train.github_text.iloc[i]) != str:
            i += 1
        self.text_vect_len = len(df_X_train.github_text.iloc[i].split())

        return self

    def transform(self, df_X):
        """
        Fill missing text values. Adds columns to track missing values.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.

        X_github : numpy.array
            Github text vectors as array.

        X_meetup : numpy.array
            Meetup text vectors as array.
        """

        df_X['github_text_missing'] = df_X['github_text'].apply(
                                        lambda x: type(x) != str)
        df_X['meetup_text_missing'] = df_X['meetup_text'].apply(
                                        lambda x: type(x) != str)
        df_X['text_missing'] = df_X['github_text_missing'] | \
                               df_X['meetup_text_missing']

        fill_val = [0] * self.text_vect_len
        X_github = np.array([map(int, x.split())
                                if type(x) == str
                                else fill_val
                                for x
                                in df_X['github_text']])
        X_meetup = np.array([map(int, x.split())
                                if type(x) == str
                                else fill_val
                                for x
                                in df_X['meetup_text']])

        df_X.drop(['github_text', 'meetup_text'], axis=1, inplace=True)

        return (df_X, X_github, X_meetup)



class text_idf(UD_transform_class):
    """
    Transforms text vectors by inverse document frequency.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------

    """


    def __init__(self, idf='yes'):

        self.params = {'idf': idf}
        self.valids = {'idf': ['yes', 'no', 'both']}

    def fit(self, (df_X_train, X_train_github, X_train_meetup), y=None):
        """
        Fit TfidfTransformers to train data.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        X_train_github : numpy.array
            Github text vectors as array.

        X_train_meetup : numpy.array
            Meetup text vectors as array.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.params['idf'] in ['yes', 'both']:
            self.tfidf_github = TfidfTransformer().fit(X_train_github)
            self.tfidf_meetup = TfidfTransformer().fit(X_train_meetup)

        return self

    def transform(self, (df_X, X_github, X_meetup)):
        """
        Transform text vectors by inverse document frequency.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        X_github : numpy.array
            Github text vectors as array.

        X_meetup : numpy.array
            Meetup text vectors as array.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.

        X_github : scipy.sparse.csr_matrix, None
            Github text vectors as sparse array, if idf = no or both else None.

        X_meetup : scipy.sparse.csr_matrix, None
            Meetup text vectors as sparse array, if idf = no or both else None.

        X_github_tfidf : scipy.sparse.csr_matrix, None
            Github text vectors with idf transform applied as sparse array,
            if idf = yes or both else None.

        X_meetup_tfidf : scipy.sparse.csr_matrix, None
            Meetup text vectors with idf transform applied as sparse array,
            if idf = yes or both else None.
        """

        if self.params['idf'] == 'no':
            return (df_X, csr_matrix(X_github), csr_matrix(X_meetup), None,
                                                                      None)

        X_github_tfidf = self.tfidf_github.transform(X_github)
        X_meetup_tfidf = self.tfidf_meetup.transform(X_meetup)

        if self.params['idf'] == 'yes':
            return (df_X, None, None, X_github_tfidf, X_meetup_tfidf)

        return (df_X, csr_matrix(X_github), csr_matrix(X_meetup),
                X_github_tfidf, X_meetup_tfidf)


class text_aggregate(UD_transform_class):
    """
    Aggregates github and meetup text vectors into similarity metrics.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    refill_missing : bool
        Indicates if similarities are refilled where text vectors were missing.

    cosine_only : bool
        Indicates if only cosine is used.

    drop_missing_bools : bool
        Indicates if missing value dummies are dropped.
    """


    def __init__(self, refill_missing=False, cosine_only=True,
                 drop_missing_bools=True):

        self.params = {'refill_missing': refill_missing,
                       'cosine_only': cosine_only,
                       'drop_missing_bools': drop_missing_bools}
        self.valids = {'refill_missing': [True, False],
                       'cosine_only': [True, False],
                       'drop_missing_bools': [True, False]}

    def fit(self, (df_X, X_github, X_meetup,
                   X_github_tfidf, X_meetup_tfidf), y=None):
        """
        Compute refill values if refill_missing = True.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input train data.

        X_github : scipy.sparse.csr_matrix, None
            Github text vectors as sparse array, if idf = no or both else None.

        X_meetup : scipy.sparse.csr_matrix, None
            Meetup text vectors as sparse array, if idf = no or both else None.

        X_github_tfidf : scipy.sparse.csr_matrix, None
            Github text vectors with idf transform applied as sparse array,
            if idf = yes or both else None.

        X_meetup_tfidf : scipy.sparse.csr_matrix, None
            Meetup text vectors with idf transform applied as sparse array,
            if idf = yes or both else None.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.params['refill_missing']:

            if X_github is not None:

                df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                      in zip(X_github, X_meetup)]

                if not self.params['cosine_only']:
                    df_X['text_norm_github'] = [norm(x) for x in X_github]
                    df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
                    df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                             df_X['text_norm_meetup']
                    df_X['text_dot'] = [x1.dot(x2.T)[0,0]
                                        for x1, x2
                                        in zip(X_github, X_meetup)]

            if X_github_tfidf is not None:

                df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0]
                                          for x1, x2
                                          in zip(X_github_tfidf,
                                                 X_meetup_tfidf)]

                if not self.params['cosine_only']:
                    df_X['text_norm_github_tfidf'] = [norm(x) for x
                                                      in X_github_tfidf]
                    df_X['text_norm_meetup_tfidf'] = [norm(x) for x
                                                      in X_meetup_tfidf]
                    df_X['DIFF:text_norm_tfidf'] = \
                        df_X['text_norm_github_tfidf'] - \
                        df_X['text_norm_meetup_tfidf']
                    df_X['text_dot_tfidf'] = [x1.dot(x2.T)[0,0]
                                              for x1, x2
                                              in zip(X_github_tfidf,
                                                     X_meetup_tfidf)]

            self.refill_cols = ['text_sim',
                                'text_sim_tfidf',
                                'text_norm_github',
                                'text_norm_meetup',
                                'DIFF:text_norm',
                                'text_norm_github_tfidf',
                                'text_norm_meetup_tfidf',
                                'DIFF:text_norm_tfidf',
                                'text_dot',
                                'text_dot_tfidf']
            self.refill_means = {}
            for col in self.refill_cols:
                if col in df_X.columns:
                    self.refill_means[col] = df_X[col] \
                                            [~ df_X['text_missing']].mean()

        return self

    def transform(self, (df_X, X_github, X_meetup, X_github_tfidf,
                         X_meetup_tfidf)):
        """
        Aggregate github and meetup text vectors into similarity metrics.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        X_github : scipy.sparse.csr_matrix, None
            Github text vectors as sparse array, if included.

        X_meetup : scipy.sparse.csr_matrix, None
            Meetup text vectors as sparse array, if included.

        X_github_tfidf : scipy.sparse.csr_matrix, None
            Github text vectors with idf transform applied as sparse array,
            if included.

        X_meetup_tfidf : scipy.sparse.csr_matrix, None
            Meetup text vectors with idf transform applied as sparse array,
            if included

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.
        """

        if X_github is not None:

            df_X['text_sim'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                in zip(X_github, X_meetup)]

            if not self.params['cosine_only']:
                df_X['text_norm_github'] = [norm(x) for x in X_github]
                df_X['text_norm_meetup'] = [norm(x) for x in X_meetup]
                df_X['DIFF:text_norm'] = df_X['text_norm_github'] - \
                                         df_X['text_norm_meetup']
                df_X['text_dot'] = [x1.dot(x2.T)[0,0] for x1, x2
                                    in zip(X_github, X_meetup)]

        if X_github_tfidf is not None:

            df_X['text_sim_tfidf'] = [cosine_similarity(x1, x2)[0,0] for x1, x2
                                      in zip(X_github_tfidf, X_meetup_tfidf)]

            if not self.params['cosine_only']:
                df_X['text_norm_github_tfidf'] = [norm(x) for x
                                                  in X_github_tfidf]
                df_X['text_norm_meetup_tfidf'] = [norm(x) for x
                                                  in X_meetup_tfidf]
                df_X['DIFF:text_norm_tfidf'] = \
                    df_X['text_norm_github_tfidf'] - \
                    df_X['text_norm_meetup_tfidf']
                df_X['text_dot_tfidf'] = [x1.dot(x2.T)[0,0] for x1, x2
                                          in zip(X_github_tfidf,
                                                 X_meetup_tfidf)]

        if self.params['refill_missing']:
            for col in self.refill_cols:
                if col in df_X.columns:
                    df_X[col][df_X['text_missing']] = self.refill_means[col]

        missing_bool_cols = ['github_text_missing',
                             'meetup_text_missing',
                             'text_missing']
        if self.params['drop_missing_bools']:
            df_X.drop(missing_bool_cols, axis=1, inplace=True)
        else:
            for col in missing_bool_cols:
                df_X[col] = df_X[col].apply(int)

        return df_X


class name_similarity(UD_transform_class):
    """
    Computes similarity between github and meetup profile names.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    use : str
        Indicates which of fullname, firstname and lastname, or calculated
        similarity to use.
    """

    def __init__(self, use='first_last'):

        self.params = {'use': use}
        self.valids = {'use': ['full', 'first_last', 'calc']}

    def transform(self, df_X):
        """
        Compute similarity between github and meetup profile names.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.
        """

        if self.params['use'] != 'first_last':
            df_X.drop(['firstname_similarity', 'lastname_similarity'],
                      axis=1, inplace=True)

        if self.params['use'] != 'full':
            df_X.drop(['fullname_similarity'], axis=1, inplace=True)

        if self.params['use'] == 'calc':
            df_X['name_sim'] = df_X.apply(lambda row: match(row['github_name'],
                                                            row['meetup_name']
                                                           ),
                                          axis=1)

        df_X.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

        return df_X


class scaler(UD_transform_class):
    """
    Standardizes data by mean and standard deviation of respective features.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    None
    """


    def fit(self, df_X_train, y=None):
        """
        Fit StandardScaler to train data.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        self.ss = StandardScaler().fit(df_X_train.values)

        return self

    def transform(self, df_X):
        """
        Standardize data by mean and standard deviation of respective features.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        df_X : pandas.DataFrame
            Transformed data.
        """

        scaled_arr = self.ss.transform(df_X.values)
        df_X = pd.DataFrame(scaled_arr, columns=df_X.columns)

        return df_X


class df_to_array(UD_transform_class):
    """
    Converts DataFrame to array.

    User-defined scikit-learn style fit/transform class.

    Parameters
    ----------
    None
    """


    def fit(self, df_X_train, y=None):
        """
        Store columns of DataFrame as feature labels.

        Parameters
        ----------
        df_X_train : pandas.DataFrame
            Input train data.

        y : list, optional
            Target labels.

        Returns
        -------
        self : object
            Returns self.
        """

        self.feats = df_X_train.columns

        return self

    def transform(self, df_X):
        """
        Convert DataFrame to array.

        Parameters
        ----------
        df_X : pandas.DataFrame
            Input data.

        Returns
        -------
        X : numpy.array
            Transformed data.
        """

        X = df_X.values
        return X
