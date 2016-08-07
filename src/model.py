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
from UD_transforms import *
from sklearn.pipeline import Pipeline
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


def threshold_acc_prec_rec(y_test, y_test_prob, mod, start, shard):
    """"""

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

    fname = str(int(start - 1470348265)).zfill(7) + '_'
    if shard:
        fname = 'shard_' + fname

    plt.savefig('../img/%sthresh_acc_prec_rec_%s' % (fname,
                                                     mod.__class__.__name__))
    plt.close('all')


def feature_importances(mod, feats, start, shard):
    """"""

    n_feats = len(feats)

    if mod.__class__ == XGBClassifier:
        fscores = mod.booster().get_fscore()
        imps = np.zeros(n_feats)
        for k, v in fscores.iteritems():
            imps[int(k[1:])] = v
    else:
        imps = mod.feature_importances_

    feats_imps = zip(feats, imps)
    feats_imps.sort(key=operator.itemgetter(1), reverse=True)
    feats = [feat for feat, imp in feats_imps]
    imps = [imp for feat, imp in feats_imps]

    print mod.__class__.__name__
    print 'FEATURE IMPORTANCES'
    for feat_imp in feats_imps:
        print str_feat_imp(feat_imp)

    fname = str(int(start - 1470348265)).zfill(7) + '_'
    if shard:
        fname = 'shard_' + fname

    fig = plt.figure(figsize=(15, 12))
    x_ind = np.arange(n_feats)
    plt.barh(x_ind, imps[::-1]/imps[0], height=(10./n_feats), align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, feats[::-1], fontsize=14)
    plt.title('%s Feature Importances' % mod.__class__.__name__)
    plt.savefig('../img/%sfeature_importances_%s' %
                (fname, mod.__class__.__name__))
    plt.close('all')

    return feats_imps


def check_duplicates(best_pred, \
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

    positive_class_ix = np.argwhere(class_arr == 1)[0,0]
    probs = estimator.predict_proba(df_X)[:, positive_class_ix]

    return probs


def filtered_predict(estimator, df_X_test, y_test=None,
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


def str_params((key, val)):
    """"""

    return ('  ' + key + ':').ljust(40) + str(val)


def str_feat_imp((feat, imp)):
    """"""
    
    return ('  ' + feat + ':').ljust(40) + str(imp)


def str_eval((key, val)):
    """"""

    return ('  ' + key + ':').ljust(50) + (str(val) if type(val) == int \
                                                else ('%.1f%%' % (val * 100)))


def best_transform(best_pipe, df_X):
    """"""

    data = df_X

    for step in best_pipe.steps:
        data = step[1].transform(data)

    return data


def save_scatter(best_pipe, df_X_train, df_X_test, y_train, y_test, start, shard):
    """"""

    df_train = best_transform(best_pipe, df_X_train)
    df_test = best_transform(best_pipe, df_X_test)
    df_train['match'] = y_train
    df_test['match'] = y_test

    fname = str(int(start - 1470348265)).zfill(7) + '_'
    if shard:
        fname = 'shard_' + fname

    scatter_matrix(df_X_train_trans, alpha=0.2, figsize=(15,12))
    plt.savefig('../img/%sscatter-matrix_train' % fname)
    plt.close('all')
    scatter_matrix(df_X_test_trans, alpha=0.2, figsize=(15,12))
    plt.savefig('../img/%sscatter-matrix_test' % fname)
    plt.close('all')


def model(df_clean, shard=False, short=False, tune=False):
    """"""

    start = start_time('Modeling...')

    y = df_clean.pop('match').values

    df_X_train, df_X_test, y_train, y_test = train_test_split(df_clean, y,
                                                test_size=0.5, random_state=0)
    print '\n# Train Obs:', len(y_train)
    print '    Counts:', np.unique(y_train, return_counts=True)
    print '# Test Obs: ', len(y_test)
    print '    Counts:', np.unique(y_test, return_counts=True)

    # suppress warning writing on copy warning
    pd.options.mode.chained_assignment = None  # default='warn'

    if short:
        mods = [XGBClassifier(seed=0)]
        gs_param_grid = [{'name_similarity__fullname': [True, False],
                          'name_similarity__lastname': [True, False]}]

    elif tune:

        mods = [XGBClassifier(seed=0)]
        gs_param_grid = [{'mod__max_depth': range(3, 10, 2),
                          'mod__min_child_weight': range(1, 6, 2),
                          'mod__gamma': [i / 10.0 for i in range(0, 5)],
                          'mod__subsample': [i/10.0 for i in range(6,10)],
                          'mod__colsample_bytree': [i/10.0 for i in range(6,10)],
                          'mod__reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
                          'dist_fill_missing__fill_with': ['median'],
                          'dist_diff__include': ['ignore_min'],
                          'text_idf__idf': ['yes'],
                          'text_aggregate__refill_missing': [True],
                          'text_aggregate__cosine_only': [True],
                          'text_aggregate__drop_missing_bools': [False],
                          'name_similarity__fullname': [True],
                          'name_similarity__firstname': [True],
                          'name_similarity__lastname': [True],
                          'name_similarity__calc': [False]}]

    else:
        mods = [XGBClassifier(seed=0),
                RandomForestClassifier(oob_score=True,
                                       n_jobs=-1,
                                       random_state=0,
                                       n_estimators=250),
                LogisticRegression(random_state=0,
                                   n_jobs=-1),
                GradientBoostingClassifier(n_estimators=250,
                                           random_state=0),
                AdaBoostClassifier(random_state=0),
                SVC(random_state=0,
                    probability=True),
                ]
        gs_param_grid = [{'dist_fill_missing__fill_with':
                            [#'mean',
                             'median',
                             #'min',
                             #'max'
                             ],
                         'dist_diff__include':
                            ['all',
                             'none',
                             'ignore_min'],
                         'text_idf__idf':
                            ['yes',
                             'no',
                             'both'],
                         'text_aggregate__refill_missing':
                            [True,
                             False],
                         'text_aggregate__cosine_only':
                            [True,
                             False],
                         'text_aggregate__drop_missing_bools':
                            [True,
                             False],
                         'name_similarity__fullname':
                            [True,
                             False],
                         'name_similarity__firstname':
                            [True,
                             False],
                         'name_similarity__lastname':
                            [True,
                             False],
                         'name_similarity__calc':
                            [True,
                             False]}]

    best_accuracy = 0.
    for mod in mods:

        print '\n'
        print mod.__class__.__name__

        df_X_train_copy = df_X_train.copy()
        df_X_test_copy = df_X_test.copy()

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
                            cv=5)

        grid.fit(df_X_train_copy, y_train)

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
        threshold_acc_prec_rec(y_test, y_test_prob, mod, start, shard)

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

        for kvpair in evals.iteritems():
            print str_eval(kvpair)

        feats_imps = feature_importances(
                        grid.best_estimator_.named_steps['mod'],
                        grid.best_estimator_.named_steps['df_to_array'].feats,
                        start, shard)

        fname = str(int(start - 1470348265)).zfill(7) + '_' \
                + mod.__class__.__name__
        if shard:
            fname = 'shard_' + fname

        with open('../output/%s.txt' % fname, 'w') as f:
            f.write('ALGORITHM\n')
            f.write(mod.__class__.__name__)

            f.write('\n\nBEST SCORE\n')
            f.write(str(grid.best_score_))

            f.write('\n\nBEST PARAMS\n')
            for kvpair in grid.best_params_.iteritems():
                f.write(str_params(kvpair))
                f.write('\n')

            f.write('\n\nMETRICS\n')
            for kvpair in evals.iteritems():
                f.write(str_eval(kvpair))
                f.write('\n')

            f.write('\n\nFEATURE IMPORTANCES\n')
            for feat_imp in feats_imps:
                f.write(str_feat_imp(feat_imp))
                f.write('\n')

            f.write('\n\nGRID SCORES\n')
            for score in grid.grid_scores_:
                f.write(str(score))
                f.write('\n')
        
        if evals['Test Accuracy (Filtered + Train)'] > best_accuracy:

            best_accuracy = evals['Test Accuracy (Filtered + Train)']

            best_mod = mod
            best_grid = grid
            best_params = grid.best_params_
            best_pipe = grid.best_estimator_
            best_evals = evals
            best_filtered_pred = filtered_predict(grid, df_X_test, y_test,
                                                  filter_train=True,
                                                  df_X_train=df_X_train,
                                                  y_train=y_train)

    check_duplicates(best_filtered_pred, github_train, meetup_train, \
                                         github_test, meetup_test)
    save_scatter(best_pipe, df_X_train, df_X_test, y_train, y_test, start, 
                 shard)
    
    end_time(start)


if __name__ == '__main__':

    read = 'read' in argv
    shard = 'shard' in argv
    short = 'short' in argv
    tune = 'tune' in argv

    if read:
        if shard:
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

    model(df_clean, shard, short, tune)
