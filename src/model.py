# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from load import *
from clean import *
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
                             GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from UD_pipe import UD_pipe
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            roc_auc_score
from copy import deepcopy
from collections import OrderedDict
import operator
import seaborn
import pdb
from sys import argv
import ast


def check_duplicates(best_pred, X_train, X_test, y_train,
                     github_train, meetup_train, github_test, meetup_test):
    """"""

    print 'There are %d train observations' % len(y_train)
    print 'There are %d test observations' % len(best_pred)
    print 'There are %d train matches' % sum(y_train)
    print 'There are %d predicted matches' % sum(best_pred)

    ix_matches_train = np.argwhere(y_train == 1).flatten()
    ix_matches_pred = np.argwhere(best_pred == 1).flatten()

    github_train_matches = github_train[ix_matches_train]
    meetup_train_matches = meetup_train[ix_matches_train]
    github_pred_matches = github_test[ix_matches_pred]
    meetup_pred_matches = meetup_test[ix_matches_pred]
    github_train_and_pred_matches = np.append(github_train_matches,
                                              github_pred_matches)
    meetup_train_and_pred_matches = np.append(meetup_train_matches,
                                              meetup_pred_matches)

    def count_duplicates(matches, set_name):

        unique_counts = np.unique(matches, return_counts=True)[1]
        duplicates = sum(unique_counts) - len(unique_counts)
        print 'There are %d duplicates in %s' % (duplicates, set_name)


    count_duplicates(github_train_matches, 'github_train_matches')
    count_duplicates(meetup_train_matches, 'meetup_train_matches')
    count_duplicates(github_pred_matches, 'github_pred_matches')
    count_duplicates(meetup_pred_matches, 'meetup_pred_matches')
    count_duplicates(github_train_and_pred_matches,
                     'github_train_and_pred_matches')
    count_duplicates(meetup_train_and_pred_matches,
                     'meetup_train_and_pred_matches')

    uniques, counts = np.unique(github_pred_matches, return_counts=True)
    duplicate_githubs = [g for g, c in zip(uniques, counts) if c > 1]
    duplicate_githubs_meetups = []
    for g in duplicate_githubs:
        ix = np.argwhere((github_test == g) & (best_pred == 1)).flatten()
        ms = meetup_test[ix]
        duplicate_githubs_meetups.append((g, ms))

    print '\ngithubs linked to multiple meetups'
    for k, v in duplicate_githubs_meetups:
        print 'github:', k
        print 'meetups:', v

    uniques, counts = np.unique(meetup_pred_matches, return_counts=True)
    duplicate_meetups = [m for m, c in zip(uniques, counts) if c > 1]
    duplicate_meetups_githubs = []
    for m in duplicate_meetups:
        ix = np.argwhere((meetup_test == m) & (best_pred == 1)).flatten()
        gs = github_test[ix]
        duplicate_meetups_githubs.append((m, gs))

    print '\nmeetups linked to multiple githubs'
    for k, v in duplicate_meetups_githubs:
        print 'meetup:', k
        print 'githubs:', v


def predict_proba_positive(estimator, df_X):
    """"""

    if hasattr(estimator, 'classes_'):
        class_arr = np.array(estimator.classes_)
    else:
        class_arr = np.array(estimator.best_estimator_.classes_)

    positive_class_ix = class_arr[class_arr == 1][0]
    probs = estimator.predict_proba(df_X)[:, positive_class_ix]

    return probs


def filtered_predict(estimator, df_X_test,
                        filter_train=False, df_X_train=None, y_train=None):
    """"""

    github_test = df_X_test['github'].values
    meetup_test = df_X_test['github'].values

    if filter_train:
        github_train = df_X_train['github'].values
        meetup_train = df_X_train['meetup'].values

        taken_githubs = set(github_train[y_train == 1])
        taken_meetups = set(meetup_train[y_train == 1])
    else:
        taken_githubs = set()
        taken_meetups = set()

    probs = predict_proba_positive(estimator, df_X_test)
    preds = np.zeros(len(probs))

    for ix in np.argsort(probs)[::-1]:
        if (probs[ix] < 0.5):
            break
        if (github_test[ix] not in taken_githubs) and \
           (meetup_test[ix] not in taken_meetups):
            preds[ix] = 1
            taken_githubs.add(github_test[ix])
            taken_meetups.add(meetup_test[ix])

    return preds


def filtered_accuracy(estimator, df_X_input, y):
    """"""

    preds = filtered_predict(estimator, df_X_input)
    accuracy = accuracy_score(y, preds)

    return accuracy


def acc_prec_rec(estimator, df_X_test, y_test, filtered=True, \
                 filter_train=False, df_X_train=None, y_train=None):
    """"""

    if filtered:
        preds = filtered_predict(estimator,
                                 df_X_test,
                                 filter_train=filter_train,
                                 df_X_train=df_X_train,
                                 y_train=y_train)
    else:
        preds = estimator.predict(df_X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    return accuracy, precision, recall


def model(df_clean, write=False, accuracy_only=False):
    """"""

    start = start_time('Modeling...')

    y = df_clean.pop('match').values

    if write:
        scatter_matrix(df_clean, alpha=0.2, figsize=(15,12))
        plt.savefig('../img/scatter_matrix_data')
        plt.close('all')

    df_X_train, df_X_test, y_train, y_test = train_test_split(df_clean, y,
                                                test_size=0.5, random_state=0)
    print '\n# Train Obs:', len(y_train)
    print '    Counts:', np.unique(y_train, return_counts=True)
    print '# Test Obs: ', len(y_test)
    print '    Counts:', np.unique(y_test, return_counts=True)

    # suppress warning writing on copy warning
    pd.options.mode.chained_assignment = None  # default='warn'

    mods = [#RandomForestClassifier(oob_score=True,
            #               n_jobs=-1,
            #               random_state=0,
            #               n_estimators=250),
            #LogisticRegression(random_state=0,
            #                   n_jobs=-1),
            #GradientBoostingClassifier(n_estimators=250,
            #                           random_state=0),
            #AdaBoostClassifier(random_state=0),
            #SVC(random_state=0,
            #    probability=True),
            XGBClassifier(seed=0)
            ]

    best_accuracy = 0.
    for mod in mods:

        print '\n'
        print mod.__class__.__name__

        df_X_train_copy = df_X_train.copy()
        df_X_test_copy = df_X_test.copy()

        # short grid for testing
        grid = GridSearchCV(estimator=UD_pipe(mod),
                            param_grid=[{}],
                            scoring=filtered_accuracy,
                            n_jobs=-1)

        """ FULL PARAM GRID
        grid = GridSearchCV(estimator=UD_pipe(mod),
                            param_grid=[{'dist_fill_with':
                                            ['mean',
                                             'median',
                                             'min',
                                             'max'],
                                         'dist_diffs':
                                            ['all',
                                             'none'],
                                         'idf':
                                            ['yes',
                                             'no',
                                             'both'],
                                         'text_refill_missing':
                                            [True,
                                             False],
                                         'text_drop_missing_bools':
                                            [True,
                                             False],
                                         'fullname':
                                            [True,
                                             False],
                                         'firstname':
                                            [True,
                                             False],
                                         'lastname':
                                            [True,
                                             False],
                                         'calc':
                                            [True,
                                             False]
                                        }
                                       ],
                            scoring=filtered_accuracy,
                            n_jobs=-1)"""

        grid.fit(df_X_train_copy, y_train)
        acc = grid.score(df_X_test_copy, y_test)

        y_train_pred = grid.predict(df_X_train_copy)
        y_test_pred = grid.predict(df_X_test_copy)

        evals = OrderedDict()
        evals['Test Accuracy'], \
            evals['Test Precision'], \
            evals['Test Recall'] = acc_prec_rec(grid,
                                                df_X_test_copy,
                                                y_test,
                                                filtered=False)

        y_test_prob = predict_proba_positive(grid, df_X_test_copy)
        evals['Test AUC'] = roc_auc_score(y_test, y_test_prob)

        evals['Test Accuracy (Filtered)'], \
            evals['Test Precision (Filtered)'], \
            evals['Test Recall (Filtered)'] = acc_prec_rec(grid,
                                                           df_X_test_copy,
                                                           y_test,
                                                           filtered=True)

        evals['Test Accuracy (Filtered + Train)'], \
            evals['Test Precision (Filtered + Train)'], \
            evals['Test Recall (Filtered + Train)'] \
                = acc_prec_rec(grid, df_X_test_copy, y_test, filtered=True,
                               filter_train=True, df_X_train=df_X_train,
                               y_train=y_train)

        for key, value in evals.iteritems():
            print ('  ' + key + ':').ljust(50), \
                    value if type(value) == int else ('%.1f%%' % (value * 100))

        fname = str(int(time()) - 1470348265).zfill(7)
        with open('../output/%s.txt' % fname, 'w') as f:
            f.write('ALGORITHM\n')
            f.write(mod.__class__.__name__)

            f.write('\n\nBEST SCORE\n')
            f.write(str(grid.best_score_))

            f.write('\n\nBEST PARAMS\n')
            f.write(str(grid.best_params_))

            f.write('\n\nMETRICS\n')
            f.write(str(evals))

            f.write('\n\nGRID SCORES\n')
            f.write(str(grid.grid_scores_))

            f.write('\n\n')

    #check_duplicates(best_pred, X_train, X_test, y_train,
                     #github_train, meetup_train, github_test, meetup_test)
    #check_duplicates(best_filtered_pred, X_train, X_test, y_train,
                     #github_train, meetup_train, github_test, meetup_test)


        """
        DELETE:

        X_train = df_X_train_trans.values
        X_test = df_X_test_trans.values

        if write:
            scatter_matrix(df_X_train_trans, alpha=0.2, figsize=(15,12))
            plt.savefig('../img/scatter_matrix_data_trans-train')
            plt.close('all')
            scatter_matrix(df_X_test_trans, alpha=0.2, figsize=(15,12))
            plt.savefig('../img/scatter_matrix_data_trans-test')
            plt.close('all')
        """

        """
        acc, prec, recall
        IMPLEMENT FOR FINAL MODEL?

        if write and (mod.__class__.__name__ == 'RandomForestClassifier'):
            thresholds = y_test_prob.copy()
            thresholds.sort()
            thresh_acc = []
            thresh_prec = []
            thresh_rec = []
            for threshold in thresholds:
                y_pred = []
                for ytp in y_test_prob:
                    y_pred.append(int(ytp >= threshold))
                thresh_acc.append(accuracy_score(y_test, y_pred))
                thresh_prec.append(precision_score(y_test, y_pred))
                thresh_rec.append(recall_score(y_test, y_pred))
            plt.plot(thresholds, thresh_acc, label='accuracy')
            plt.plot(thresholds, thresh_prec, label='precision')
            plt.plot(thresholds, thresh_rec, label='recall')
            plt.legend()
            plt.savefig('../img/performance')
            plt.close('all')
        """

        """
        feats = df_X_train_trans.columns
        evals['# Features'] = len(feats)
        num_feats_plot = evals['# Features']

        if hasattr(mod, 'feature_importances_'):
            imps = mod.feature_importances_
            feats_imps = zip(feats, imps)
            feats_imps.sort(key=operator.itemgetter(1), reverse=True)
            feats = []
            imps = []
            useless_feats = []
            for feat, imp in feats_imps:
                feats.append(feat)
                imps.append(imp)
                if imp == 0:
                    useless_feats.append(feat)
            evals['# Useless Features'] = len(useless_feats)
            print mod.__class__.__name__
            print 'FEATURE IMPORTANCES'
            for feat, imp in feats_imps:
                print feat.ljust(30), imp / feats_imps[0][1]

        if write and (mod.__class__.__name__ == 'RandomForestClassifier'):
            fig = plt.figure(figsize=(15, 12))
            x_ind = np.arange(num_feats_plot)
            plt.barh(x_ind, imps[num_feats_plot-1::-1]/imps[0], height=.3,
                     align='center')
            plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
            plt.yticks(x_ind, feats[num_feats_plot-1::-1], fontsize=14)
            plt.title('RFC Feature Importances')
            plt.savefig('../img/feature_importances')
            plt.close('all')
        """

    end_time(start)
    pdb.set_trace()


if __name__ == '__main__':

    if 'read' in argv:
        if 'shard' in argv:
            df_clean = pd.read_csv('../data/clean_shard.csv')
        else:
            ints_in_argv = [int(arg) for arg in argv if arg.isdigit()]
            if ints_in_argv:
                rows = ints_in_argv[0]
                df_clean = pd.read_csv('../data/clean.csv', nrows=rows)
            else:
                df_clean = pd.read_csv('../data/clean.csv')
    else:
        df_clean = clean(load())

    write = 'write' in argv
    accuracy_only = 'evals' not in argv

    model(df_clean, write, accuracy_only)
