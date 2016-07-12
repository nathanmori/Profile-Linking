'''
NOTES

  LINKS:


  1. BUSINESS UNDERSTANDING
      Formulate question/goal

   Data Engineering
'''

import pandas as pd
from pandas.tools.plotting import scatter_matrix as scat_mat
import matplotlib.pyplot as plt
import pdb

def read_data():
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

def df_summary():
    plt.close('all')
    for i, df in enumerate(dfs):
        print '\n\n\n\ndf:', (str(i) + ' ') * 40
        print '\nshape:', df.shape
        print '\ncolumns:', df.columns
        print '\n', df.info()
        print '\n', df.head()
        print '\ni:', i
        plot = raw_input("Enter 'y' to plot, 'q' to quit >")
        if plot == 'y':
            scat_mat(df, alpha=df.shape[0]**-.5, figsize=(15, 9))
            plt.show()
        elif plot == 'q':
            break

'''
dfs[0]  (7756, 21)      login, search, message, booking data

dfs[1]  (420, 5)        counts by age_bucket, country_destination, gender
dfs[2]  (10, 7)         country info (lat, long, distance, area, language)
dfs[3]  (62096, 2)      user countries
dfs[4]  (10567737, 6)   user actions (lookup, click), device_type, secs_elapsed
dfs[5]  (62096, 15)     user profile and activity info.  All accounts created between 7/1/14 and 9/30/14
dfs[6]  (213451, 16)    similar to dfs[5], plus country_destination
'''

#pd.value_counts(df0.id_visitor)
#print df0[column].head()
