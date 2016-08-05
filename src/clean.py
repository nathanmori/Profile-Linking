# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import pdb
import numpy as np
import pandas as pd
from sys import argv
from load import *


def check_nulls(df):
    """"""

    for col in df.columns:
        print df[col].isnull().value_counts()


def ix_nulls(df):
    """"""

    print np.argwhere(df.isnull().values)


def clean(df):
    """"""

    start = start_time('Cleaning...')

    df_clean = df.copy()
    df_clean['match'] = df['correct_match'].apply(int)
    df_clean.drop(['correct_match'], axis=1, inplace=True)
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

    if 'write' in argv:

        if 'shard' in argv:

            ints_in_argv = [int(arg) for arg in argv if arg.isdigit()]
            rows = ints_in_argv[0] if ints_in_argv else 100
            df_clean.head(rows).to_csv('../data/clean_shard.csv', index=False,\
                                       encoding='utf-8')
                                       
        else:

            df_clean.to_csv('../data/clean.csv', index=False, encoding='utf-8')
