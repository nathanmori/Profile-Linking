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
    '''
    ]
#select count(*), count(id), count(deserialized_text_document_vector) from github_users;
#select count(*), count(id), count(deserialized_text_document_vector) from meetup_users;

dfs = []
for query in queries:
    c.execute(query)
    data = c.fetchall()
    cols = [desc[0] for desc in c.description]
    df = pd.DataFrame(data, columns=cols)
    dfs.append(df)
df = dfs[0]

c.execute('select id from github_users')
github_ids = c.fetchall()
c.execute('select id from meetup_users')
meetup_ids = c.fetchall()

conn.commit()
conn.close()


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
for col in cols:
    print '\n', col
    print 'all:', sum(df[col].value_counts()), 'out of', df.shape[0]
    print 'matches:', sum(df[df[split_col] == True][col].value_counts()), \
            'out of', df[split_col].value_counts()[True]
    print 'not matches:', sum(df[df[split_col] == False][col].value_counts()), \
            'out of', df[split_col].value_counts()[False]
# NOTE: face_pics exists for 971 out of 44534 (18 / 1111 matches, 953 / 43423 not matches)
# NOTE: face_pics_processed === face_pics_matched === False when exists
df.drop(['face_pics_processed', 'face_pics_matched'], axis=1, inplace=True)

# NOTE: profile_pics_matched === verified === correct_match
df.drop(['verified', 'correct_match'], axis=1, inplace=True)

# NOTE: github_meetup_combined is missing values.  Could easily create, but not valuable
df.drop(['github_meetup_combined'], axis=1, inplace=True)


matches = df[df['profile_pics_matched']]

print '# unique githubs in matches:', len(matches.github.value_counts())
print '# unique meetups in matches:', len(matches.meetup.value_counts())
print '# total githubs in matches:', sum(matches.github.value_counts())
print '# total meetups in matches:', sum(matches.meetup.value_counts())

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

print '# github taken by matches:', sum(df.github.apply(lambda x: x in github_taken))
print '# meetup taken by matches:', sum(df.meetup.apply(lambda x: x in meetup_taken))

print '# github or meetup taken by matches', sum(df.apply(lambda x:
                                            x['github'] in github_taken or
                                            x['meetup'] in meetup_taken
                                            , axis=1))

print '# github or meetup taken by matches but not a match',\
        sum(df.apply(lambda x: (    x['github'] in github_taken
                                    or
                                    x['meetup'] in meetup_taken)
                               and
                               x['profile_pics_matched'] == False
                                            , axis=1))

not_matches = df[df.apply(lambda x: (x['github'] in github_taken
                                        or
                                        x['meetup'] in meetup_taken)
                                    and
                                    x['profile_pics_matched'] == False,
                            axis=1)]

print '# unique githubs in not_matches:', len(not_matches.github.value_counts())
print '# unique meetups in not_matches:', len(not_matches.meetup.value_counts())
print '# total githubs in not_matches:', sum(not_matches.github.value_counts())
print '# total meetups in not_matches:', sum(not_matches.meetup.value_counts())
