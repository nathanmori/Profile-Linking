# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
from sys import argv
from load import *


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

    end_time(start)

    return df_clean


if __name__ == '__main__':
    
    if 'read' in argv:
        if 'shard' in argv:
            df = pd.read_csv('../data/similars_shard.csv') 
        else:
            df = pd.read_csv('../data/similars.csv')
    else:
        df = load()

    df_clean = clean(df)
