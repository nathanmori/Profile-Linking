# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from load import *
from clean import *
from engr import *
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
                            roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import seaborn
import pdb
import sys
from sys import argv


class premodel(object):

    def __init__(self):
        pass

    def fit(self, df_X_train):
        pass

    def transform(self, df_X):
        pass


def print_evals(model_name, evals):
    
    print model_name
    for key, value in evals:
        print ('  ' + key + ':').ljust(25), \
                    value if type(value) == int else ('%.1f%%' % (value * 100))


def model(df_engr, write=False):

    start = start_time('Modeling...')

    df_copy = df_engr.copy()
    y = df_copy.pop('match').values

    """
    Plot scatter matrix.
    """
    if write:
        scatter_matrix(df_copy, alpha=0.2, figsize=(15,12))
        plt.savefig('../img/scatter_matrix')
        plt.close('all')

    """
    Train/test split.
    """
    df_X_train, df_X_test, y_train, y_test = train_test_split(df_copy, y,
                                                test_size=0.5, random_state=0)

    # suppress warning (that changes to df_X_train and df_X_test won't make it
    # back to df_copy
    pd.options.mode.chained_assignment = None  # default='warn'

    """
    GLOBAL FIT / TRANSFORM
    (model that applies to all algorithms, so performed once up front)

    MISSING VALUES IN DISTANCE COLUMNS
    FILLING WITH MEAN - CONSIDER LATER

    later: all four distances missing at 6 index locations:
        [ 717] [1409] [1433] [1463] [1694] [1986]
    could drop, but assuming we want model to be able to handle,
        will fill with mean

    PLOT HIST WITH AND WITHOUT FILLING"""
    dist_cols = ['min_dist_km', 'avg_dist_km', 'median_dist_km', 'max_dist_km']
    dist_means = df_X_train[dist_cols].dropna().apply(lambda ser: ser.apply(float)).mean()
    df_X_train.loc[:, dist_cols] = df_X_train[dist_cols].fillna(dist_means).apply(lambda ser: ser.apply(float))
    df_X_test.loc[:, dist_cols] = df_X_test[dist_cols].fillna(dist_means).apply(lambda ser: ser.apply(float))

    """CONVERT TO MODEL CLASS....FIT:"""
    X_train_github = np.array(df_X_train['github_text'].apply(lambda x: x.flatten().tolist()).tolist())
    X_train_meetup = np.array(df_X_train['meetup_text'].apply(lambda x: x.flatten().tolist()).tolist())
    tfidf_github = TfidfTransformer()
    tfidf_meetup = TfidfTransformer()
    X_train_github_tfidf = tfidf_github.fit_transform(X_train_github)
    X_train_meetup_tfidf = tfidf_meetup.fit_transform(X_train_meetup)
    """TRANSFORM:"""
    X_test_github = np.array(df_X_test['github_text'].apply(lambda x: x.flatten().tolist()).tolist())
    X_test_meetup = np.array(df_X_test['meetup_text'].apply(lambda x: x.flatten().tolist()).tolist())
    X_test_github_tfidf = tfidf_github.transform(X_test_github)
    X_test_meetup_tfidf = tfidf_meetup.transform(X_test_meetup)

    df_X_train['text_sim'] = [cosine_similarity(x1, x2) for x1, x2 in
                              zip(X_train_github_tfidf, X_train_meetup_tfidf)]
    df_X_test['text_sim'] = [cosine_similarity(x1, x2) for x1, x2 in 
                             zip(X_test_github_tfidf, X_test_meetup_tfidf)]
    df_X_train.drop(['github_text', 'meetup_text'], axis=1, inplace=True)
    df_X_test.drop(['github_text', 'meetup_text'], axis=1, inplace=True)

    X_train = df_X_train.values
    X_test = df_X_test.values

    """
    Model.
    """
    mod = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=0, \
                                 n_estimators=250).fit(X_train, y_train)

    """
    Calculate predictions (using 0.5 threshold) and probabilities of test data.
    """
    y_train_pred = mod.predict(X_train)
    y_test_pred = mod.predict(X_test)
    y_test_prob = mod.predict_proba(X_test)[:,1]

    """
    Calculate scores of interest (using threshold of 0.5).
    """
    evals = []
    evals.append(('Train Accuracy', mod.score(X_train, y_train)))
    evals.append(('Train Precision', precision_score(y_train, y_train_pred)))
    evals.append(('Train Recall', recall_score(y_train, y_train_pred)))
    evals.append(('OOB Accuracy', mod.oob_score_))
    evals.append(('Test Accuracy', mod.score(X_test, y_test)))
    evals.append(('Test Precision', precision_score(y_test, y_test_pred)))
    evals.append(('Test Recall', recall_score(y_test, y_test_pred)))
    evals.append(('Test AUC', roc_auc_score(y_test, y_test_prob)))

    """
    Calculate accuracy, precision, recall for varying thresholds.
    """
    if write:
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
    num_feats_plot = min(15, df_copy.shape[1])
    feats = df_copy.columns
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
    evals.append(('# Features', len(feats)))
    evals.append(('# Useless Features', len(useless_feats)))

    if write:
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

    print_evals('Random Forest', evals)

    return mod


if __name__ == '__main__':

    write = True if 'write' in argv else False

    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
    mod = model(df_engr, write)
