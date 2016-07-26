import psycopg2
from datetime import datetime
import pdb
from copy import deepcopy
import os
import pandas as pd

params = {
  'database': os.environ['TALENTFUL_PG_DATABASE'],
  'user': os.environ['TALENTFUL_PG_USER'],
  'password': os.environ['TALENTFUL_PG_PASSWORD'],
  'host': os.environ['TALENTFUL_PG_HOST'],
  'port': os.environ['TALENTFUL_PG_PORT']}
conn = psycopg2.connect(**params)
c = conn.cursor()
queries = [
    '''
        select
            *
        from
            github_meetup_staging
    ''',
    '''
        select
            id
            , deserialized_text_document_vector
        from
            github_users
        limit
            10
    ''',
    '''
        select
            id
            , deserialized_text_document_vector
        from
            meetup_users
        limit
            10
    ''',
    '''
        select
            count(*)
            , count(id)
            , count(deserialized_text_document_vector)
        from
            github_users
    ''',
    '''
        select
            count(*)
            , count(id)
            , count(deserialized_text_document_vector)
        from
            meetup_users
    ''',
    '''
    select
        id
    from
        github_users
    ''',
    '''
    select
        id
    from
        meetup_users
    '''
    ]
dfs = []
for query in queries:
    c.execute(query)
    data = c.fetchall()
    cols = [desc[0] for desc in c.description]
    df = pd.DataFrame(data, columns=cols)
    dfs.append(df)

conn.commit()
conn.close()

df, df_gh_head, df_mu_head, df_gh_counts, df_mu_counts, github_users, meetup_users = dfs

# NOTE: id is a unique and meaningless identifier
df.drop(['id'], axis=1, inplace=True)

# NOTE: name_similar, profile_pics_processed === True`
# NOTE: randomly_paired === False
df.drop(['name_similar', 'profile_pics_processed', 'randomly_paired'], axis=1, inplace=True)

# NOTE: profile_pics vs face_pics?
"""
NEED TO DISCUSS
"""
### NOTE: EDA ###
cols = ['face_pics_processed', 'face_pics_matched']
split_col = 'profile_pics_matched'
# NOTE: face_pics exists for 971 out of 44534 (18 / 1111 matches, 953 / 43423 not matches)
# NOTE: face_pics_processed === face_pics_matched === False when exists
df.drop(['face_pics_processed', 'face_pics_matched'], axis=1, inplace=True)

# NOTE: profile_pics_matched === verified === correct_match
df.drop(['verified', 'correct_match'], axis=1, inplace=True)

# NOTE: github_meetup_combined is missing values.  Could easily create, but not valuable
df.drop(['github_meetup_combined'], axis=1, inplace=True)


matches = df[df['profile_pics_matched']]

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


not_matches = df[df.apply(lambda x: (x['github'] in github_taken
                                        or
                                        x['meetup'] in meetup_taken)
                                    and
                                    x['profile_pics_matched'] == False,
                            axis=1)]
