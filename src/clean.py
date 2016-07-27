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
    df.drop(['github', 'meetup'], axis=1, inplace=True)
    df['profile_pics_matched'] = df['profile_pics_matched'].apply(int)
    """
    CONVERT github_name AND meetup_name TO name_similarity
    """
    """
    CONVERT github_text AND meetup_text (STRINGS) TO VECTORS
    """

    """
    MISSING VALUES IN DISTANCE COLUMNS
    FILLING WITH MEAN - CONSIDER LATER

    PLOT HIST WITH AND WITHOUT FILLING
    """
    df.fillna(df.mean(), inplace=True)

    return df



if __name__ == '__main__':
    df = load_data()
    df_clean = clean(df)
