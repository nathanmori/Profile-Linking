###################################
# NOTES
#
#   LINKS:
#   https://www.reddit.com/r/Myfitnesspal/comments/3jwofg/how_do_i_view_other_peoples_food_diaries/
#   https://www.reddit.com/r/loseit/comments/32cqai/myfitnesspal_users_who_share_their_diaries/
#
#
#   1. BUSINESS UNDERSTANDING
#       Formulate question/goal
#
#   Data Engineering
#

import pandas as pd
from pandas.tools.plotting import scatter_matrix as scat_mat
import matplotlib.pyplot as plt
import pdb
plt.close('all')

#data1
dfs = [pd.read_csv('data1/airbnb_session_data.txt', sep='|')]
locs = ['data2/age_gender_bkts.csv',
        'data2/countries.csv',
        'data2/sample_submission_NDF.csv',
        'data2/sessions.csv',
        'data2/test_users.csv',
        'data2/train_users.csv']
#data2
for loc in locs:
    dfs.append(pd.read_csv(loc))

for i, df in enumerate(dfs):
    print '\n\n\n\ndf:', (str(i) + ' ') * 40
    print 'shape:', df.shape
    print 'columns:', df.columns
    print df.info()
    print df.head()
    pdb.set_trace()
    scat_mat(df, alpha=df.shape[0]**-.5, figsize=(15, 9))
    plt.show()
'''
dfs[0]
dfs[1]: counts by age_bucket, country_destination, gender

'''

'''
pd.value_counts(df0.id_visitor)
print df0[column].head()'''
