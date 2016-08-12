# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from load import *
from clean import *
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_apr_vs_thresh(y_test, y_test_prob, mod, start, shard):
    """
    Plot accuracy, precision, and recall vs. probability threshold for positive
    class.

    Parameters
    ----------
    y_test : list
        Target labels of test data.

    y_test_prob : list
        Predicted probabilities of positive class for test data.

    mod : object
        Estimator used.

    start : float
        Time at start of run.

    shard : bool
        Indicates if shard of data is used.

    Returns
    -------
    None
    """

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

    fig = plt.figure(figsize=(20, 12))
    plt.plot(thresholds, thresh_acc, label='accuracy')
    plt.plot(thresholds, thresh_prec, label='precision')
    plt.plot(thresholds, thresh_rec, label='recall')
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=32)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=32)
    plt.title('%s Accuracy, Precision, Recall (Unfiltered) vs. Threshold' %
              mod.__class__.__name__, fontsize=40)
    plt.legend(fontsize=32, loc='lower left')
    fig.tight_layout()

    fname = str(int(start - 1470348265)).zfill(7) + '_'
    if shard:
        fname = 'shard_' + fname
    plt.savefig('../output/%sthresh_acc_prec_rec_%s' %
                (fname, mod.__class__.__name__))
    plt.close('all')


def feature_importances(mod, feats, start, shard, write):
    """
    Extract and plot feature importances.

    Parameters
    ----------
    mod : object
        Estimator used.

    feats : list
        Features used.

    start : float
        Time at start of run.

    shard : bool
        Indicates if shard of data is used.

    write : bool
        Indicates if plots are to be written.

    Returns
    -------
    feats_imps : list of tuples
        Features and importances.
    """

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

    if write:

        fig = plt.figure(figsize=(20, 12))
        x_ind = np.arange(n_feats)
        plt.barh(x_ind, imps[::-1]/imps[0], height=.8, align='center')
        plt.ylim(x_ind.min() - .5, x_ind.max() + .5)
        plt.yticks(x_ind, feats[::-1], fontsize=32)
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=32)
        plt.title('%s Feature Importances' % mod.__class__.__name__,
                  fontsize=40)
        plt.gcf().tight_layout()

        fname = str(int(start - 1470348265)).zfill(7) + '_'
        if shard:
            fname = 'shard_' + fname
        plt.savefig('../output/%sfeature_importances_%s' %
                    (fname, mod.__class__.__name__))
        plt.close('all')

    return feats_imps


def count_duplicates(matches, set_name):
    """
    Count and report number of duplicates (i.e. githubs linked to multiple
    meetups and vice versa).

    Parameters
    ----------
    matches : numpy.array
        Set of ids associated with matches.

    set_name : str
        Name of set.

    Returns
    -------
    None
    """

    unique_counts = np.unique(matches, return_counts=True)[1]
    duplicates = sum(unique_counts) - len(unique_counts)
    print 'There are %d duplicates in %s' % (duplicates, set_name)


def check_duplicates(best_pred, github_train, meetup_train, github_test,
                     meetup_test):
    """
    Check predictions for duplicate githubs and meetups linked to the same
    individual and report.

    Parameters
    ----------
    best_pred : list
        Predicted classes.

    github_train : numpy.array
        Train dataset github ids.

    meetup_train : numpy.array
        Train dataset meetup ids.

    github_test : numpy.array
        Test dataset github ids.

    meetup_test : numpy.array
        Test dataset meetup ids.

    Returns
    -------
    None
    """

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
    """
    Extract predicted probabilities of positive class from estimator.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X : pandas.DataFrame
        Input data.

    Returns
    -------
    probs : numpy.array
        Predicted probabilities of positive class.
    """

    if hasattr(estimator, 'classes_'):
        class_arr = np.array(estimator.classes_)
    else:
        class_arr = np.array(estimator.best_estimator_.classes_)

    positive_class_ix = np.argwhere(class_arr == 1)[0, 0]
    probs = estimator.predict_proba(df_X)[:, positive_class_ix]

    return probs


def filtered_predict(estimator, df_X_test, y_test=None,
                     filter_train=False, df_X_train=None, y_train=None):
    """
    Predict class labels with filtering.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_test : pandas.DataFrame
        Input test data.

    y_test : list, optional
        Test target labels.

    filter_train : bool, default=False
        Indicates if train data is included in filtering.

    df_X_train : pandas.DataFrame
        Input train data, not used if filter_train=False.

    y_train : list, optional
        Train target labels, not used if filter_train=False.

    Returns
    -------
    preds : numpy.array
        Filtered class predictions.
    """

    github_test = df_X_test['github'].values
    meetup_test = df_X_test['meetup'].values

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


def prob_predict(unfiltered_probs, github_test, meetup_test, duplicate_ixs,
                 taken_githubs, taken_meetups):
    """
    Calculates probabilities of alternatives for a profile with duplicate
    usage. Functions recursively, calling self for following duplicates.

    Parameters
    ----------
    unfiltered_probs : list-like
        Probabilities of test observations being positive as calculated by the
        estimator before filtering is applied.

    github_test : list-like
        Github ids of test observations.

    meetup_test : list-like
        Meetup ids of test observations.

    duplicate_ixs : list of lists
        Contains one sublist for each github or meetup profile used
        more than once. Each sublist contains indexes of all pairs in which
        the profile is used.

    taken_githubs : set
        Contains github profile ids that have been used so far.

    taken_meetups : set
        Contains meetup profile ids that have been used so far.

    Returns
    -------
    probability : float
        Probability of predictions being correct.

    predictions : list of lists
        List of lists corresponding to duplicate_ixs. Each sublist contains one
        1 (the positive) and all other values 0 (negatives).
    """

    if not duplicate_ixs:
        return None, None

    pair_ixs = duplicate_ixs[0]
    pair_probs = []
    pair_preds = []

    if len(duplicate_ixs) == 1:

        for pair_ix in pair_ixs:
            if (github_test[pair_ix] in taken_githubs) and \
                    (meetup_test[pair_ix] in taken_meetups):

                pair_probs.append(1)
                pair_preds.append([])

            elif (github_test[pair_ix] in taken_githubs) or \
                    (meetup_test[pair_ix] in taken_meetups):

                pair_probs.append(0)
                pair_preds.append(None)

            else:

                pair_probs.append(unfiltered_probs[pair_ix])
                pair_preds.append([])

    else:

        for pair_ix in pair_ixs:
            taken_githubs_copy = deepcopy(taken_githubs)
            taken_githubs_copy.add(github_test[pair_ix])
            taken_meetups_copy = deepcopy(taken_meetups)
            taken_meetups_copy.add(meetup_test[pair_ix])
            pair_prob, pair_pred = \
                prob_predict(unfiltered_probs, github_test, meetup_test,
                             duplicate_ixs[1:], taken_githubs_copy,
                             taken_meetups_copy)
            pair_preds.append(pair_pred)

            if (github_test[pair_ix] in taken_githubs) and \
                    (meetup_test[pair_ix] in taken_meetups):

                pair_probs.append(1)

            elif (github_test[pair_ix] in taken_githubs) or \
                    (meetup_test[pair_ix] in taken_meetups):

                pair_probs.append(0)

            else:

                pair_probs.append(pair_prob * unfiltered_probs[pair_ix])

    best_pair_arg = np.argmax(pair_probs)
    best_pair_ix = pair_ixs[best_pair_arg]

    probability = pair_probs[best_pair_arg]

    predictions = pair_preds[best_pair_arg]
    predictions.insert(0, [0] * len(pair_ixs))
    predictions[0][best_pair_arg] = 1

    return probability, predictions


def recursive_filtered_predict(estimator, df_X_test, y_test=None,
                               filter_train=False, df_X_train=None,
                               y_train=None):
    """
    Predict class labels with filtering, accounting for overall probability.
    Calls recursive prob_predict to evaluate all possible predictions.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_test : pandas.DataFrame
        Input test data.

    y_test : list, optional
        Test target labels.

    filter_train : bool, default=False
        Indicates if train data is included in filtering.

    df_X_train : pandas.DataFrame
        Input train data, not used if filter_train=False.

    y_train : list, optional
        Train target labels, not used if filter_train=False.

    Returns
    -------
    preds : numpy.array
        Filtered class predictions.
    """

    github_test = df_X_test['github'].values
    meetup_test = df_X_test['meetup'].values

    if filter_train:
        github_train = df_X_train['github'].values
        meetup_train = df_X_train['meetup'].values

        taken_githubs = set(github_train[y_train == 1])
        taken_meetups = set(meetup_train[y_train == 1])
    else:
        taken_githubs = set()
        taken_meetups = set()

    unfiltered_probs = predict_proba_positive(estimator, df_X_test)
    positive_githubs = github_test[unfiltered_probs >= 0.5]
    positive_meetups = meetup_test[unfiltered_probs >= 0.5]

    unique_githubs, count_githubs = np.unique(positive_githubs,
                                              return_counts=True)
    unique_meetups, count_meetups = np.unique(positive_meetups,
                                              return_counts=True)

    duplicate_githubs = unique_githubs[count_githubs > 1]
    duplicate_meetups = unique_meetups[count_meetups > 1]

    duplicate_ixs = []
    for github in duplicate_githubs:
        duplicate_ixs.append(np.argwhere([(x[0] == github) and (x[1] >= 0.5)
                                          for x in
                                          zip(github_test.tolist(),
                                              unfiltered_probs.tolist())]
                                         ).flatten().tolist())
    for meetup in duplicate_meetups:
        duplicate_ixs.append(np.argwhere([(x[0] == meetup) and (x[1] >= 0.5)
                                          for x in
                                          zip(meetup_test.tolist(),
                                              unfiltered_probs.tolist())]
                                         ).flatten().tolist())

    prob, duplicate_preds = prob_predict(unfiltered_probs, github_test,
                                         meetup_test, duplicate_ixs,
                                         taken_githubs, taken_meetups)

    preds = [round(x) for x in unfiltered_probs]
    for i, pair_ixs in enumerate(duplicate_ixs):
        for j, pair_ix in enumerate(pair_ixs):
            preds[pair_ix] = duplicate_preds[i][j]

    return preds


def filtered_accuracy(estimator, df_X_input, y):
    """
    Compute filtered accuracy.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_input : pandas.DataFrame
        Input data.

    y : list
        Target labels.

    Returns
    -------
    accuracy : float
        Filtered accuracy.
    """

    preds = filtered_predict(estimator, df_X_input)
    accuracy = accuracy_score(y, preds)

    return accuracy


def filtered_roc(estimator, df_X_test, y_test, filter_train=False,
                 df_X_train=None, y_train=None, plot=True, return_FPRs=False,
                 return_TPRs=False):
    """
    Compute and plot ROC curve.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_test : pandas.DataFrame
        Input test data.

    y_test : list
        Test target labels.

    filter_train : bool, default=False
        Indicates if train data is included in filtering.

    df_X_train : pandas.DataFrame
        Input train data, not used if filter_train=False.

    y_train : list
        Train target labels, not used if filter_train=False.

    plot : bool, default=True
        Indicates if ROC is plotted.

    return_FPRs : bool, default=False
        Indicates if false positive rates are returned.

    return_TPRs : bool, default=False
        Indicates if true positive rates are returned.

    Returns
    -------
    FPRs : list
        False positive rates, returned if return_FPRs=True.

    TPRs : list
        True positive rates, returned if return_TPRs=True.
    """

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

    TPRs = [0.]
    FPRs = [0.]
    n_obs = len(y_test)

    for ix in np.argsort(probs)[::-1]:
        if (github_test[ix] not in taken_githubs) and \
           (meetup_test[ix] not in taken_meetups):
            preds[ix] = 1
            taken_githubs.add(github_test[ix])
            taken_meetups.add(meetup_test[ix])

        TPs = sum(np.multiply(preds, y_test))
        Ps = float(sum(y_test))
        FPs = sum(preds) - TPs
        Ns = float(len(y_test) - sum(y_test))

        TPRs.append(TPs / Ps)
        FPRs.append(FPs / Ns)

    TPRs.append(1.)
    FPRs.append(1.)

    if plot:
        plot_label = 'Test - Filtered%s' % (' + Train' if filter_train else '')
        plt.plot(FPRs, TPRs, label=plot_label)

    if return_FPRs or return_TPRs:
        returns = []
        if return_FPRs:
            returns.append(FPRs)
        if return_TPRs:
            returns.append(TPRs)

        return tuple(returns)


def filtered_roc_auc_score(estimator, df_X_test, y_test, filter_train=False,
                           df_X_train=None, y_train=None):
    """
    Compute filtered AUC.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_test : pandas.DataFrame
        Input test data.

    y_test : list
        Test target labels.

    filter_train : bool, default=False
        Indicates if train data is included in filtering.

    df_X_train : pandas.DataFrame
        Input train data, not used if filter_train=False.

    y_train : list
        Train target labels, not used if filter_train=False.

    Returns
    -------
    AUC : float
        Area under ROC curve using filtered predictions.
    """

    FPRs, TPRs = filtered_roc(estimator, df_X_test, y_test, filter_train,
                              df_X_train, y_train, plot=False,
                              return_FPRs=True, return_TPRs=True)

    AUC = 0.
    for i in xrange(len(TPRs) - 1):
        AUC += (TPRs[i + 1] + TPRs[i]) / 2 * (FPRs[i + 1] - FPRs[i])

    return AUC


def acc_prec_rec(estimator, df_X_test, y_test, filtered=True,
                 filter_train=False, df_X_train=None, y_train=None):
    """
    Compute accuracy, precision, and recall.

    Parameters
    ----------
    estimator : object
        Estimator used.

    df_X_test : pandas.DataFrame
        Input test data.

    y_test : list
        Test target labels.

    filtered : bool, default=True
        Indicates if filtering is used.

    filter_train : bool, default=False
        Indicates if train data is included in filtering, not used if
        filtered=False.

    df_X_train : pandas.DataFrame
        Input train data, not used if filtered=False or filter_train=False.

    y_train : list
        Train target labels, not used if filtered=False or filter_train=False.

    Returns
    -------
    accuracy : float
        Test accuracy of estimator.

    precision : float
        Test precision of estimator.

    recall : float
        Test recall of estimator.
    """

    if filtered:
        if filter_train:
            preds = filtered_predict(estimator,
                                     df_X_test,
                                     filter_train=filter_train,
                                     df_X_train=df_X_train.copy(),
                                     y_train=y_train)
        else:
            preds = filtered_predict(estimator, df_X_test, y_test)
    else:
        preds = estimator.predict(df_X_test)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    return accuracy, precision, recall


def str_params((key, val)):
    """
    Convert parameter name and value to string with consistent spacing.

    Parameters
    ----------
    key : str
        Name of parameter.

    val : any
        Value of parameter

    Returns
    -------
    param_line : str
        String of key: val with consistent spacing.
    """

    param_line = ('  ' + key + ':').ljust(40) + str(val)

    return param_line


def str_feat_imp((feat, imp)):
    """
    Convert feature and importance to string with consistent spacing.

    Parameters
    ----------
    feat : str
        Name of feature.

    imp : float
        Importance of feature.

    Returns
    -------
    feat_imp_line : str
        String of feat: imp with consistent spacing.
    """

    feat_imp_line = ('  ' + feat + ':').ljust(40) + str(imp)

    return feat_imp_line


def str_eval((key, val)):
    """
    Convert evaluation metric name and value to string with consistent spacing.

    Parameters
    ----------
    key : str
        Name of metric.

    val : float
        Value of metric.

    Returns
    -------
    eval_line : str
        String of key: val with consistent spacing.
    """

    eval_line = (('  ' + key + ':').ljust(50) +
                 (str(val) if type(val) == int else ('%.1f%%' % (val * 100))))

    return eval_line


def pipe_transform(best_pipe, df_X, return_array=False):
    """
    Perform pipeline transformations except for final estimator's predict.

    Parameters
    ----------
    best_pipe : object
        Pipeline used.

    df_X : pandas.DataFrame
        Input data.

    return_array : bool, default=False
        Indicates if data is returned as an array (else DataFrame).

    Returns
    -------

    data : pandas.DataFrame or numpy.array
        Transformed data.
    """

    data = df_X

    for step in best_pipe.steps[:(-1 if return_array else -2)]:
        data = step[1].transform(data)

    return data


def save_scatter(best_pipe, df_X, y, start, shard):
    """
    Plots and saves scatter_matrix of data.

    Parameters
    ----------
    best_pipe : object
        Pipeline used.

    df_X : pandas.DataFrame
        Input data.

    y : list
        Target labels.

    start : float
        Time at start of run.

    shard : bool
        Indicates if shard of data is used.

    Returns
    -------
    None
    """

    df = pipe_transform(best_pipe, df_X)
    df['match'] = y

    colors = ['red', 'green']
    scatter_matrix(df, alpha=0.25, figsize=(20, 12),
                   c=df.match.apply(lambda x: colors[x]))

    fname = str(int(start - 1470348265)).zfill(7) + '_'
    if shard:
        fname = 'shard_' + fname
    plt.savefig('../output/%sscatter-matrix' % fname)
    plt.close('all')


def model(df_clean, shard=False, short=False, tune=False, final=False,
          write=False):
    """
    Perform core modeling tasks, referencing other functions and classes.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        Input data.

    shard : bool, default=False
        Indicates if shard of data is used.

    short : bool, default=False
        Indicates if 'short' list of algorithms, features, and tuning
        parameters is used.

    tune : bool, default=False
        Indicates if 'tune' list of algorithms, features, and tuning
        parameters is used.

    final : bool, default=False
        Indicates if 'final' list of algorithms, features, and tuning
        parameters is used.

    write : bool, default=False
        Indicates if plots are to be written.

    Returns
    -------
    None
    """

    start = start_time('Modeling...')

    y = df_clean.pop('match').values

    df_X_train, df_X_test, y_train, y_test = train_test_split(df_clean, y,
                                                              test_size=0.5,
                                                              random_state=0)
    print '\n# Train Obs:', len(y_train)
    print '    Counts:', np.unique(y_train, return_counts=True)
    print '# Test Obs: ', len(y_test)
    print '    Counts:', np.unique(y_test, return_counts=True)

    # suppress warning: writing on copy
    pd.options.mode.chained_assignment = None  # default='warn'

    all_mods = [XGBClassifier(seed=0, gamma=0.1, colsample_bytree=0.75),
                RandomForestClassifier(n_jobs=-1, random_state=0),
                LogisticRegression(random_state=0, n_jobs=-1, C=0.5),
                GradientBoostingClassifier(n_estimators=250, random_state=0),
                AdaBoostClassifier(random_state=0, learning_rate=0.1),
                SVC(random_state=0, probability=True)]

    best_feats = {'dist_fill_missing__fill_with': ['median'],
                  'dist_diff__diffs': ['none'],
                  'dist_diff__keep': ['median'],
                  'text_idf__idf': ['yes'],
                  'text_aggregate__refill_missing': [True],
                  'text_aggregate__cosine_only': [True],
                  'text_aggregate__drop_missing_bools': [True],
                  'name_similarity__use': ['first_last']}
    all_feats = {'dist_fill_missing__fill_with': ['mean', 'median', 'min',
                                                  'max'],
                 'dist_diff__diffs': ['none', 'range'],
                 'dist_diff__keep': ['min', 'avg', 'median', 'max'],
                 'text_idf__idf': ['yes', 'no', 'both'],
                 'text_aggregate__refill_missing': [True, False],
                 'text_aggregate__cosine_only': [True, False],
                 'text_aggregate__drop_missing_bools': [True, False],
                 'name_similarity__use': ['full', 'first_last', 'calc']}

    if short:
        mods = all_mods
        gs_param_grid = best_feats

    elif tune:
        mods = [all_mods[4], all_mods[2], all_mods[0]]
        mod_tune = {XGBClassifier: [{'mod__max_depth': [3, 4],
                                     # default = 3
                                     # best = 3
                                     'mod__min_child_weight': [1, 2],
                                     # default = 1
                                     # best = 1
                                     'mod__gamma': [0, 0.1],
                                     # default = 0
                                     # best = 0.1
                                     'mod__subsample': [0.75, 1],
                                     # default = 1
                                     # best = 1
                                     'mod__colsample_bytree': [0.75, 1]
                                     # default = 1
                                     # best = 0.75
                                     }],

                    LogisticRegression: [{'mod__penalty': ['l2', 'l1'],
                                          # default = 'l2'
                                          # best = 'l2'
                                          'mod__C': [0.1, 0.5, 1],
                                          # default = 1.0
                                          # best = 0.5
                                          'mod__solver': ['liblinear']
                                          # default = 'liblinear'
                                          # best = 'liblinear'
                                          },
                                         {'mod__penalty': ['l2'],
                                          'mod__C': [0.1, 0.5, 1],
                                          'mod__solver': ['sag']}],

                    AdaBoostClassifier: [{'mod__learning_rate':
                                          [0.01, 0.1, 0.5, 1]
                                          # default = 1.0
                                          # best = 0.1
                                          }]}
        for key, val in mod_tune.iteritems():
            mod_tune[key] = []
            for vd in val:
                vd.update(best_feats)
                mod_tune[key].append(vd)

    elif final:
        mods = [all_mods[4]]
        gs_param_grid = {'mod__learning_rate': [0.1]}
        gs_param_grid.update(best_feats)

    else:
        mods = all_mods
        gs_param_grid = all_feats

    best_accuracy = 0.
    for mod in mods:

        if tune:
            gs_param_grid = mod_tune[mod.__class__]

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

        grid.fit(df_X_train.copy(), y_train)

        y_test_pred = grid.predict(df_X_test.copy())

        evals = OrderedDict()
        evals['Test Accuracy'], \
            evals['Test Precision'], \
            evals['Test Recall'] = acc_prec_rec(grid,
                                                df_X_test.copy(),
                                                y_test,
                                                filtered=False)

        y_test_prob = predict_proba_positive(grid, df_X_test.copy())
        evals['Test AUC'] = roc_auc_score(y_test, y_test_prob)

        evals['Test Accuracy (Filtered)'], \
            evals['Test Precision (Filtered)'], \
            evals['Test Recall (Filtered)'] = acc_prec_rec(grid,
                                                           df_X_test.copy(),
                                                           y_test,
                                                           filtered=True)
        evals['Test AUC (Filtered)'] = \
            filtered_roc_auc_score(grid, df_X_test, y_test)

        evals['Test Accuracy (Filtered + Train)'], \
            evals['Test Precision (Filtered + Train)'], \
            evals['Test Recall (Filtered + Train)'] \
            = acc_prec_rec(grid, df_X_test.copy(), y_test, filtered=True,
                           filter_train=True, df_X_train=df_X_train.copy(),
                           y_train=y_train)
        evals['Test AUC (Filtered + Train)'] = \
            filtered_roc_auc_score(grid, df_X_test, y_test, filter_train=True,
                                   df_X_train=df_X_train.copy(),
                                   y_train=y_train)

        for kvpair in evals.iteritems():
            print str_eval(kvpair)

        feat_imp_exists = (hasattr(grid.best_estimator_.named_steps['mod'],
                                   'feature_importances_') or
                           mod.__class__ == XGBClassifier)
        if feat_imp_exists:
            feats_imps = \
                feature_importances(
                    grid.best_estimator_.named_steps['mod'],
                    grid.best_estimator_.named_steps['df_to_array'].feats,
                    start, shard, write)

        df_trans = pipe_transform(grid.best_estimator_, df_clean)
        df_trans['match'] = y
        arr_corrcoef = np.corrcoef(df_trans.values.T)
        df_corrcoef = pd.DataFrame(arr_corrcoef, columns=df_trans.columns)
        df_corrcoef.index = df_corrcoef.columns

        # sklearn: known i, predicted j
        y_pred = filtered_predict(grid, df_X_test)
        arr_confusion_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
        df_confusion_matrix = pd.DataFrame(arr_confusion_matrix,
                                           columns=['Predicted Positive',
                                                    'Predicted Negative'])
        df_confusion_matrix.index = ['Actual Positive',
                                     'Actual Negative']

        fname = str(int(start - 1470348265)).zfill(7) + '_' \
            + mod.__class__.__name__
        if shard:
            fname = 'shard_' + fname
        fname = '../output/' + fname + '.txt'
        print '\nSee %s for output.\n' % fname

        with open(fname, 'w') as f:
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

            f.write('\n\nCONFUSION MATRIX\n')
            f.write(str(df_confusion_matrix))
            f.write('\n')

            if feat_imp_exists:
                f.write('\n\nFEATURE IMPORTANCES\n')
                for feat_imp in feats_imps:
                    f.write(str_feat_imp(feat_imp))
                    f.write('\n')

            f.write('\n\nCORRELATION MATRIX\n')
            f.write(str(df_corrcoef))
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
            best_prob = y_test_prob

    print 'Best Model: %s\n' % best_mod.__class__.__name__

    if write:

        save_scatter(best_pipe, df_clean, y, start, shard)

        # plot_apr_vs_thresh(y_test, best_prob, best_mod, start, shard)

        fig = plt.figure(figsize=(20, 12))
        fpr, tpr, thresholds = roc_curve(y_test, best_prob)
        plt.plot(fpr, tpr, label='Test - Unfiltered')
        filtered_roc(best_grid, df_X_test, y_test)
        filtered_roc(best_grid, df_X_test, y_test, filter_train=True,
                     df_X_train=df_X_train.copy(), y_train=y_train)
        plt.legend(fontsize=32, loc='lower right')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.2), fontsize=32)
        plt.yticks(np.arange(0, 1.1, 0.2), fontsize=32)
        plt.xlabel('False Positive Rate', fontsize=32)
        plt.ylabel('True Positive Rate', fontsize=32)
        plt.title('Best Model (%s) ROCs' % best_mod.__class__.__name__,
                  fontsize=40)
        plt.gcf().tight_layout()

        fname = str(int(start - 1470348265)).zfill(7) + '_'
        if shard:
            fname = 'shard_' + fname
        plt.savefig('../output/%sROCs' % fname)
        plt.close('all')

    end_time(start)


if __name__ == '__main__':

    read = 'read' in argv
    shard = 'shard' in argv
    short = 'short' in argv
    tune = 'tune' in argv
    final = 'final' in argv
    write = 'write' in argv

    if (short + tune + final) > 1:
        print 'COULD NOT RUN'
        print 'short, tune, and final are incompatible arguments'

    else:

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

        model(df_clean, shard, short, tune, final, write)
