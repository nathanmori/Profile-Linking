import psycopg2
from datetime import datetime
import pdb
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from load_data import load_data


def check_nulls(df):

    for col in df.columns:
        print df[col].isnull().value_counts()


def ix_nulls(df):

    print np.argwhere(df.isnull().values)


def clean(df):
    
    df_clean = df.copy()
 
    df_clean.drop(['github', 'meetup'], axis=1, inplace=True)
    df_clean['profile_pics_matched'] = df['profile_pics_matched'].apply(int)
    """
    CONVERT github_name AND meetup_name TO name_similarity
    """
    """
    CONVERT github_text AND meetup_text (STRINGS) TO VECTORS
    """

    """
    MISSING VALUES IN DISTANCE COLUMNS
    FILLING WITH MEAN - CONSIDER LATER

    later: all four distances missing at 6 index locations:
    [ 717] [1409] [1433] [1463] [1694] [1986]
    could drop, but assuming we want model to be able to handle,
    will fill with mean

    PLOT HIST WITH AND WITHOUT FILLING
    """
    dist_cols = ['min_dist_km', 'avg_dist_km', 'median_dist_km', 'max_dist_km']
    dist_means = df[dist_cols].dropna().apply(lambda ser: ser.apply(float)).mean()
    df_clean[dist_cols] = df[dist_cols].fillna(dist_means).apply(lambda ser: ser.apply(float))

    """ Convert text vectors from strings to list of ints, fill missing values with empty text
    CONSIDER DROPPING FOR TRAINING PURPOSES """ 
    text_vect_len = len(df_clean.github_text[0].split())
    df_clean.github_text = df_clean.github_text.apply(lambda x: map(int, x.split()) if x else [0] * text_vect_len)
    df_clean.meetup_text = df_clean.meetup_text.apply(lambda x: map(int, x.split()) if x else [0] * text_vect_len) 
 
    return df_clean


if __name__ == '__main__':
    df = load_data()
    df_clean = clean(df)
