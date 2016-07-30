# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

from load import *
from clean import *
from name_tools import match
import sys


def engr(df_clean):

    start = start_time('Feature engineering...')
    end_time(start)

    return df_clean


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
