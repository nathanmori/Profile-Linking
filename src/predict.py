# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from load import *
from clean import *
from model import *
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from UD_transforms import *
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            roc_auc_score, roc_curve, confusion_matrix
from copy import deepcopy
from collections import OrderedDict
import operator
import seaborn
import pdb
from sys import argv
import ast
import matplotlib
import cPickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def predict(df_knowns, df_unknowns):
    """
    Perform core modeling tasks, referencing other functions and classes.

    Parameters
    ----------
    df_knowns : pandas.DataFrame
        Input data to fit on.

    df_unknowns : pandas.DataFrame
        Input data to predict.

    Returns
    -------
    None
    """

    start = start_time('Predicting...')

    y_knowns = df_knowns.pop('match').values
    df_unknowns.pop('match')

    print '\n# Known Obs:', len(y_knowns)
    print '# Unknown Obs: ', df_unknowns.shape[0]

    # suppress warning: writing on copy
    pd.options.mode.chained_assignment = None  # default='warn'

    mod = AdaBoostClassifier(random_state=0, learning_rate=0.1)
    # change AdaBoostClassifier to whichever algorithm to use
    gs_param_grid = {'dist_fill_missing__fill_with': ['median'],
                     'dist_diff__diffs': ['none'],
                     'dist_diff__keep': ['median'],
                     'text_idf__idf': ['yes'],
                     'text_aggregate__refill_missing': [True],
                     'text_aggregate__cosine_only': [True],
                     'text_aggregate__drop_missing_bools': [True],
                     'name_similarity__use': ['first_last']}

    print '\n'
    print mod.__class__.__name__

    grid = GridSearchCV(Pipeline([('drop_github_meetup',
                                   drop_github_meetup()),
                                  ('dist_fill_missing',
                                   dist_fill_missing()),
                                  ('dist_diff',
                                   dist_diff()),
                                  ('text_fill_missing',
                                   text_fill_missing()),
                                  ('text_idf',
                                   text_idf()),
                                  ('text_aggregate',
                                   text_aggregate()),
                                  ('name_similarity',
                                   name_similarity()),
                                  ('scaler',
                                   scaler()),
                                  ('df_to_array',
                                   df_to_array()),
                                  ('mod',
                                   mod)]),
                        param_grid=gs_param_grid,
                        scoring=filtered_accuracy,
                        n_jobs=-1,
                        iid=False,
                        cv=3)

    grid.fit(df_X_known.copy(), y_known)

    unknown_predictions = filtered_predict(grid, df_unknowns,
                                           filter_train=True,
                                           df_X_train=df_knowns,
                                           y_train=y_knowns)

    end_time(start)


if __name__ == '__main__':

    read = 'read' in argv
    # reads data from .csv

    if read:
        df_knowns = pd.read_csv('../data/clean.csv')
    else:
        df_knowns = clean(load())

    df_unknowns = clean(load_unknowns())
    # NEED load_unknowns() function

    predict(df_knowns, df_unknowns)
