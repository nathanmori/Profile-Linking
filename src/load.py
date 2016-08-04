# -*- coding: utf-8 -*-

# Author: Nathan Mori <nathanmori@gmail.com>

import psycopg2
import pdb
import os
import pandas as pd
from time import time
import sys
from sys import argv


def open_conn():
    """
    Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Read more in the :ref:`User Guide <cosine_similarity>`.

    Parameters
    ----------
    X : ndarray or sparse array, shape: (n_samples_X, n_features)
        Input data.

    Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.17
           parameter *dense_output* for sparse output.

    Returns
    -------
    kernel matrix : array
        An array with shape (n_samples_X, n_samples_Y)."""

    params = {
      'database': os.environ['TALENTFUL_PG_DATABASE'],
      'user': os.environ['TALENTFUL_PG_USER'],
      'password': os.environ['TALENTFUL_PG_PASSWORD'],
      'host': os.environ['TALENTFUL_PG_HOST'],
      'port': os.environ['TALENTFUL_PG_PORT']}
    conn = psycopg2.connect(**params)
    c = conn.cursor()

    return conn, c


def close_conn(conn):

    conn.commit()
    conn.close()


def start_time(text):

    sys.stdout.write(text.ljust(25))

    return time()


def end_time(start):

    print 'DONE (%.2f seconds).' % (time() - start)


def query_to_df(query):

    #print '\nExecuting query:'
    #print ' '.join(query.split())

    conn, c = open_conn()
    c.execute(query)
    data = c.fetchall()
    cols = [desc[0] for desc in c.description]
    close_conn(conn)

    return pd.DataFrame(data, columns=cols)


def query_columns(table):

    query = 'select * from %s limit 0' % table
    conn, c = open_conn()
    c.execute(query)
    cols = [desc[0] for desc in c.description]
    close_conn(conn)

    return cols


def load():

    start = start_time('Loading data...')

    cols = query_columns('github_meetup_staging')
    # NOTE: id is a unique and meaningless identifier
    # NOTE: name_similar, profile_pics_processed === True
    # NOTE: randomly_paired === False
    # NOTE: ignore face_pics
    # NOTE: profile_pics_matched === verified === correct_match
    # NOTE: github_meetup_combined is missing values
    # NOTE: ADDING github, meetup back in to check duplicates
    # NOTE: github_meetup_combined is not significant
    drop_cols = ['id',
                 'name_similar',
                 'profile_pics_processed',
                 'randomly_paired',
                 'face_pics_processed',
                 'face_pics_matched',
                 'verified',
                 #'correct_match',  Replaced with profile_pics_matched
                 'profile_pics_matched',
                 'github_meetup_combined']#,
                 #'github',
                 #'meetup']
    keep_cols = [col for col in cols if col not in drop_cols]
    col_string = ', '.join(keep_cols)

    matches_query = '''
        with matches as
        (
            select
                github
                , meetup
            from
                github_meetup_staging
            where
                correct_match = 't'
        )
        select
            %s
            , github_users.deserialized_text_document_vector as github_text
            , meetup_users.deserialized_text_document_vector as meetup_text
        from
            github_meetup_staging as similars
            left join github_users
                on similars.github = github_users.id
            left join meetup_users
                on similars.meetup = meetup_users.id
        where
            similars.github in (select github from matches)
            or similars.meetup in (select meetup from matches)
        ''' % col_string
    df = query_to_df(matches_query)

    end_time(start)

    return df

"""
NEED TO ADDRESS DUPLICATE github, meetup IN matches
- same person? consider a match OR label for multi-profiles?
- different person? drop rows OR create separate label for non-match matches
"""

if __name__ == '__main__':

    df = load()

    if 'write' in argv:
        df.to_csv('../data/similars.csv', index=False, encoding='utf-8')
        
    if 'shard' in argv:
        ints_in_argv = [int(arg) for arg in argv if arg.isdigit()]
        rows = ints_in_argv[0] if ints_in_argv else 100
        df.head(rows).to_csv('../data/similars_shard.csv', encoding='utf-8')
