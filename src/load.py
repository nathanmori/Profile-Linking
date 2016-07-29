import psycopg2
from datetime import datetime
import pdb
from copy import deepcopy
import os
import pandas as pd
from time import time
import sys


def open_conn():

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


def start(text):

    sys.stdout.write(text.ljust(25))

    return time()


def end_time(start_time):

    print 'DONE (%.2f seconds).' % (time() - start_time)


def query_to_df(query):

    #print '\nExecuting query:'
    #print ' '.join(query.split())
    #print '...'
    #start_time = time()

    conn, c = open_conn()
    c.execute(query)
    data = c.fetchall()
    cols = [desc[0] for desc in c.description]
    close_conn(conn)

    #end_time(start_time)

    return pd.DataFrame(data, columns=cols)


def get_columns(table):

    query = 'select * from %s limit 0' % table
    conn, c = open_conn()
    c.execute(query)
    cols = [desc[0] for desc in c.description]
    close_conn(conn)

    return cols


def load():

    start_time = start('Loading data...')

    similars_cols = get_columns('github_meetup_staging')
    # NOTE: id is a unique and meaningless identifier
    # NOTE: name_similar, profile_pics_processed === True`
    # NOTE: randomly_paired === False
    # NOTE: profile_pics vs face_pics?
    """
    NEED TO DISCUSS
    """
    # NOTE: face_pics exists for 971 out of 44534 (18 / 1111 matches, 953 / 43423 not matches)
    # NOTE: face_pics_processed === face_pics_matched === False when exists
    # NOTE: profile_pics_matched === verified === correct_match
    # NOTE: github_meetup_combined is missing values.  Could easily create, but not valuable
    # NOTE: github_meetup_combined is not significant
    similars_drops = ['id',
                      'name_similar',
                      'profile_pics_processed',
                      'randomly_paired',
                      'face_pics_processed',
                      'face_pics_matched',
                      'verified',
                      'correct_match',
                      'github_meetup_combined']
    similars_keeps = [col for col in similars_cols if col not in similars_drops]
    similars_col_string = ', '.join(similars_keeps)

    matches_query = '''
        with matches as
        (
            select
                github
                , meetup
            from
                github_meetup_staging
            where
                profile_pics_matched = 't'
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
        ''' % similars_col_string
    df = query_to_df(matches_query)

    end_time(start_time)

    return df
"""
NEED TO ADDRESS DUPLICATE github, meetup IN matches
- same person? consider a match - ALSO CONSIDER ASSUMPTION BELOW
- different person? drop rows OR create separate label for non-match matches
"""

"""
NEED TO CONSIDER IF github/meetup TAKEN BY A MATCH BIASES THE RESULTS
"""

"""
ASSUMPTION: No user has multiple profiles on the same site.
"""

if __name__ == '__main__':
    load()
