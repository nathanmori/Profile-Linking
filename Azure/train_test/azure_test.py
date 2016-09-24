# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import numpy as np
import pandas as pd
from copy import deepcopy
import operator
import ast

from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import cPickle as pickle


def azureml_main(df, df_pipe_pickle):

    df_clean = clean(df)

    y = df_clean.pop('match').values

    # suppress warning: writing on copy
    pd.options.mode.chained_assignment = None  # default='warn'

    pipe_pickle = df_pipe_pickle[0][0]
    pipe = pickle.loads(pipe_pickle)
    df_trans = pipe.transform(df_clean.copy())
    df_trans['match'] = y

    return df_trans
