import psycopg2
from datetime import datetime
import pdb
from copy import deepcopy
import os
import pandas as pd
from time import time

def query_to_df(query):
    c.execute(query)
    data = c.fetchall()
    cols = [desc[0] for desc in c.description]
    return pd.DataFrame(data, columns=cols)

params = {
  'database': os.environ['TALENTFUL_PG_DATABASE'],
  'user': os.environ['TALENTFUL_PG_USER'],
  'password': os.environ['TALENTFUL_PG_PASSWORD'],
  'host': os.environ['TALENTFUL_PG_HOST'],
  'port': os.environ['TALENTFUL_PG_PORT']}
conn = psycopg2.connect(**params)
c = conn.cursor()

print '\nconnection open'
start_time = time()

print '\nloading similars...'
df_similars = query_to_df('''
                          select
                            *
                          from
                            github_meetup_staging
                          ''')

print '\nloading githubs...'
df_githubs = query_to_df('''
                         select
                           id
                           , deserialized_text_document_vector
                         from
                           github_users
                         ''')

print '\nloading meetups...'
df_meetups = query_to_df('''
                         select
                           id
                           , deserialized_text_document_vector
                         from
                           meetup_users
                         ''')

conn.commit()
conn.close()

df_similars.drop(['id'], axis=1, inplace=True)

# NOTE: name_similar, profile_pics_processed === True`
# NOTE: randomly_paired === False
df_similars.drop(['name_similar', 'profile_pics_processed', 'randomly_paired'], axis=1, inplace=True)

# NOTE: profile_pics vs face_pics?
"""
NEED TO DISCUSS
"""
### NOTE: EDA ###
cols = ['face_pics_processed', 'face_pics_matched']
split_col = 'profile_pics_matched'
# NOTE: face_pics exists for 971 out of 44534 (18 / 1111 matches, 953 / 43423 not matches)
# NOTE: face_pics_processed === face_pics_matched === False when exists
df_similars.drop(['face_pics_processed', 'face_pics_matched'], axis=1, inplace=True)

# NOTE: profile_pics_matched === verified === correct_match
df_similars.drop(['verified', 'correct_match'], axis=1, inplace=True)

# NOTE: github_meetup_combined is missing values.  Could easily create, but not valuable
df_similars.drop(['github_meetup_combined'], axis=1, inplace=True)


matches = df_similars[df_similars['profile_pics_matched']]

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
github_taken = matches.github.unique()
meetup_taken = matches.meetup.unique()


not_matches = df_similars[df_similars.apply(lambda x: (x['github'] in github_taken
                                        or
                                        x['meetup'] in meetup_taken)
                                    and
                                    x['profile_pics_matched'] == False,
                            axis=1)]

print '\nmatches and not_matches ready'
print 'time taken = %d seconds' % (time() - start_time)
