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
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import operator
import seaborn
import pdb


def model(df_engr):

    print 'Modeling...'
    start_time = time()

    df_copy = df_engr.copy()

    y = df_copy.pop('profile_pics_matched').values
    X = df_copy.values

    """
    Plot scatter matrix.
    """
    scatter_matrix(df_copy, alpha=0.2, figsize=(15,12))
    plt.savefig('scatter_matrix')
    plt.close('all')

    """
    Train/test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=0.5, random_state=0)

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
    Calculate and print scores of interest (using threshold of 0.5).
    """
    print 'RFC Train Data Score:', mod.score(X_train, y_train)
    print 'RFC Test Data Score:', mod.score(X_test, y_test)
    print 'RFC Out Of Bag Score:', mod.oob_score_
    print 'RFC Train Precision:', precision_score(y_train, y_train_pred)
    print 'RFC Test Precision:', precision_score(y_test, y_test_pred)
    print 'RFC Train Recall:', recall_score(y_train, y_train_pred)
    print 'RFC Test Recall:', recall_score(y_test, y_test_pred)
    print 'RFC AUC:', roc_auc_score(y_test, y_test_prob)

    """
    Calculate accuracy, precision, recall for varying thresholds.
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
    plt.plot(thresholds, thresh_acc, label='accuracy')
    plt.plot(thresholds, thresh_prec, label='precision')
    plt.plot(thresholds, thresh_rec, label='recall')
    plt.legend()
    plt.savefig('performance')
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
    print 'RFC num_feats:', len(feats)
    print 'RFC num_useless_feats:', len(useless_feats)
    fig = plt.figure(figsize=(15, 10))
    x_ind = np.arange(num_feats_plot)
    plt.barh(x_ind, imps[num_feats_plot-1::-1]/imps[0], height=.3, align='center')
    plt.ylim(x_ind.min() + .5, x_ind.max() + .5)
    plt.yticks(x_ind, feats[num_feats_plot-1::-1], fontsize=14)
    plt.title('RFC Feature Importances')
    plt.savefig('feature_importances')
    plt.close('all')

    end_time(start_time)

    return mod


if __name__ == '__main__':

    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
    mod = model(df_engr)
