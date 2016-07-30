# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from datetime import datetime
import pdb
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from load import *
from time import time
import sys


def check_nulls(df):

    for col in df.columns:
        print df[col].isnull().value_counts()


def ix_nulls(df):

    print np.argwhere(df.isnull().values)


def clean(df):

    start = start_time('Cleaning...')
    
    df_clean = df.copy()
    df_clean['match'] = df['profile_pics_matched'].apply(int)
    df_clean.drop(['profile_pics_matched'], axis=1, inplace=True)
    df_clean['github_name'] = df['github_name'].apply(
                                                lambda x: ' '.join(x.split()))
    df_clean['meetup_name'] = df['meetup_name'].apply(
                                                lambda x: ' '.join(x.split()))

    """ Convert text vectors from strings to list of ints, 
            fill missing values with empty text
        CONSIDER leaving empty and fill missing cosine sims with mean cosine
            sim after calc'd """
    text_vect_len = len(df_clean.github_text[0].split())
    df_clean.github_text = df_clean.github_text.apply(lambda x:
                            np.array(map(int, x.split())).reshape(1,-1)
                            if x else np.zeros((1, text_vect_len), dtype=int))
    df_clean.meetup_text = df_clean.meetup_text.apply(lambda x:
                            np.array(map(int, x.split())).reshape(1,-1)
                            if x else np.zeros((1, text_vect_len), dtype=int))

    end_time(start)
 
    return df_clean


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
