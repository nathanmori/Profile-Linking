###################################
# NOTES
#
#   LINKS:
#   https://www.reddit.com/r/Myfitnesspal/comments/3jwofg/how_do_i_view_other_peoples_food_diaries/
#   https://www.reddit.com/r/loseit/comments/32cqai/myfitnesspal_users_who_share_their_diaries/


import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pdb

df1 = pd.read_csv('data1/airbnb_session_data.txt', sep='|')
df1.head()
df1.info()
scatter_matrix(df1, alpha=0.1, figsize=(15, 9))
plt.show()
pd.value_counts(df1.id_visitor)
print df1.columns
for column in df1.columns:
    print df1[column].head()
    pd.value_counts(df1.sent_booking_request)


# In[27]:

locs = ['data2/age_gender_bkts.csv',
        'data2/countries.csv',
        'data2/sample_submission_NDF.csv',
        'data2/sessions.csv',
        'data2/test_users.csv',
        'data2/train_users.csv']

dfs = []
for loc in locs:
    dfs.append(pd.read_csv(loc))


# In[25]:

for i, df in enumerate(dfs):
    print '/n', i
    print df.shape
    print df.columns

# In[28]:

for df in dfs:
    scatter_matrix(df, alpha=.1, figsize=(15, 9), diagonal='kde')
    plt.show()
    pdb.set_trace()


# In[ ]:
