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
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from name_tools import match
from copy import deepcopy
import operator
import seaborn
import pdb
import sys
from sys import argv
import dist_fill_missing
import dist_diff
import text
import name_similarity
import scaler
import ast


def get_classes(step):
    """"""

    with open(step + '.py') as f:
        source = f.read()
    p = ast.parse(source)
    classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]

    return classes

def print_evals(model_name, evals, accuracy_only=True):
    """"""

    print '\n' + model_name
    if accuracy_only:
        print (' Test Accuracy: %.1f%%').ljust(25) % (
                                evals['Test Accuracy'] * 100)
        return

    for key, value in evals.iteritems():
        print ('  ' + key + ':').ljust(25), \
                    value if type(value) == int else ('%.1f%%' % (value * 100))

def strip_users(df_X):
    """"""

    github = df_X['github'].values
    meetup = df_X['meetup'].values
    
    df_X.drop(['github', 'meetup'], axis=1, inplace=True)

    return df_X, github, meetup


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


def filter_duplicates(prob, github_test, meetup_test, y_train=None,
                      github_train=None, meetup_train=None,
                      filter_train=False):
    """"""
    
    if filter_train:
        taken_githubs = set(github_train[y_train == 1])
        taken_meetups = set(meetup_train[y_train == 1])
    else:
        taken_githubs = set()
        taken_meetups = set()

    preds = np.zeros(len(prob))

    for ix in np.argsort(prob)[::-1]:
        if (prob[ix] < 0.5):
            break
        if (github_test[ix] not in taken_githubs) and \
           (meetup_test[ix] not in taken_meetups):
            preds[ix] = 1
            taken_githubs.add(github_test[ix])
            taken_meetups.add(meetup_test[ix])

    return preds
    

def model(df_clean, write=False, accuracy_only=True):
    """"""

    start = start_time('Modeling...')

    df_copy = df_clean.copy()
    y = df_copy.pop('match').values

    """
    Plot scatter matrix.
    """
    if write:
        scatter_matrix(df_copy, alpha=0.2, figsize=(15,12))
        plt.savefig('../img/scatter_matrix_data')
        plt.close('all')

    """
    Train/test split.
    """
    df_X_train, df_X_test, y_train, y_test = train_test_split(df_copy, y,
                                                test_size=0.5, random_state=0)

    # suppress warning (that changes to df_X_train and df_X_test won't make it
    # back to df_copy
    pd.options.mode.chained_assignment = None  # default='warn'

    df_X_train, github_train, meetup_train = strip_users(df_X_train)
    df_X_test, github_test, meetup_test = strip_users(df_X_test)

    premod = Pipeline([('dist_fill_missing', dist_fill_missing.mean()),
                       ('dist_diff', dist_diff.all()),
                       ('text', text.all()),
                       ('name_similarity', name_similarity.name_tools_match()),
                       ('scaler', scaler.standard())])
    df_X_train_fit = df_X_train.copy()
    premod.fit(df_X_train_fit)
    df_X_train_trans = premod.transform(df_X_train)
    df_X_test_trans = premod.transform(df_X_test)
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
    Model.
    """
    mods = [LogisticRegression(random_state=0,
                               n_jobs=-1),
            RandomForestClassifier(oob_score=True,
                                   n_jobs=-1,
                                   random_state=0,
                                   n_estimators=250),
            GradientBoostingClassifier(n_estimators=250,
                                       random_state=0),
            AdaBoostClassifier(random_state=0),
            GaussianNB(),
            SVC(random_state=0,
                probability=True),
            LinearSVC(random_state=0),
            NuSVC(random_state=0)]
        #Neural Network
        #Logit boosting (ada boost variant)

    best_accuracy = 0.
    for mod in mods:

        """ ADD FUNCTIONALITY TO STORE VARIABLES SEPARATELY """

        mod.fit(X_train, y_train)

        """
        Calculate predictions (using 0.5 threshold).
        """
        y_train_pred = mod.predict(X_train)
        y_test_pred = mod.predict(X_test)
        """ CONSIDER FOR CLASSES WITHOUT predict_proba """
        if hasattr(mod, 'predict_proba'):
            positive_class_ix = mod.classes_[mod.classes_ == 1][0]
            y_test_prob = mod.predict_proba(X_test)[:, positive_class_ix]

        """
        Calculate scores of interest (using threshold of 0.5).
        """
        evals = {}
        evals['Train Accuracy'] = mod.score(X_train, y_train)
        evals['Train Precision'] = precision_score(y_train, y_train_pred)
        evals['Train Recall'] = recall_score(y_train, y_train_pred)
        if hasattr(mod, 'oob_score_'):
            evals['OOB Accuracy'] = mod.oob_score_
        evals['Test Accuracy'] = mod.score(X_test, y_test)
        evals['Test Precision'] = precision_score(y_test, y_test_pred)
        evals['Test Recall'] = recall_score(y_test, y_test_pred)
        if hasattr(mod, 'predict_proba'):
            evals['Test AUC'] = roc_auc_score(y_test, y_test_prob)

        """
        Calculate accuracy, precision, recall for varying thresholds.
        """
        """ UPDATE FOR OTHER MODELS """
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
        Calculate and plot feature importances.
        Useless features from earlier runs have been removed in clean_data.
        """
        feats = df_X_train_trans.columns
        evals['# Features'] = len(feats)
        num_feats_plot = evals['# Features'] #min(15, df_copy.shape[1])
        """CONSIDER FOR ALGS W/O feature_importances_"""
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

        end_time(start)

        print_evals(mod.__class__.__name__, evals, accuracy_only)

        if callable(getattr(mod, 'predict_proba', None)):
            filtered_pred = filter_duplicates(y_test_prob, github_test,
                                      meetup_test)
            filtered_score = accuracy_score(y_test, filtered_pred)
            print ('  Filtered Test Accuracy: %.1f%%').ljust(25) % (
                                            filtered_score * 100)
            
            filtered_pred = filter_duplicates(y_test_prob, github_test,
                                              meetup_test, y_train,
                                              github_train, meetup_train,
                                              filter_train=True)
            filtered_score = accuracy_score(y_test, filtered_pred)
            print ('  Filtered (w/ train) Test Accuracy: %.1f%%').ljust(25) % (
                                            filtered_score * 100)


        if ((evals['Test Accuracy'] > best_accuracy) and
             callable(getattr(mod, 'predict_proba', None))):

            best_accuracy = deepcopy(evals['Test Accuracy'])
            best_mod = deepcopy(mod)
            best_pred = deepcopy(y_test_pred)
            best_prob = deepcopy(y_test_prob)
            best_filtered_pred = deepcopy(filtered_pred)

    print '\n\nBest Model with predict_proba:', best_mod.__class__.__name__
    print 'Test Accuracy: %.1f%%' % (100. * best_accuracy)

    check_duplicates(best_pred, X_train, X_test, y_train,
                     github_train, meetup_train, github_test, meetup_test)
    check_duplicates(best_filtered_pred, X_train, X_test, y_train,
                     github_train, meetup_train, github_test, meetup_test)
    
    return best_mod, X_test


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
    #filter_train = 'filter_train' in argv

    best_mod, X_test = model(df_clean, write, accuracy_only)
