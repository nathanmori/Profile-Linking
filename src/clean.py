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
    np.argwhere(df.isnull().values)


def clean(df):
    pass

if __name__ == '__main__':
    df = load_data()
    df_clean = clean(df)
