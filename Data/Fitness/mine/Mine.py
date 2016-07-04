import pandas as pd
import numpy as np
from datetime import date as dt
import matplotlib.pyplot as plt
import pdb

#read data
fname = 'Fitness.xlsx'
df = pd.read_excel(fname, header=1)

#clean data
df.reset_index(inplace=True)
df.dropna(1, how='all', inplace=True)
cols_orig = df.columns

df_colnums = pd.DataFrame(np.array(range(df.shape[1])).reshape(1,-1), columns=df.columns).append(df, ignore_index=True)
cols_drop = df_colnums.columns[[0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27, 32, 33, 34, 35]]
df_colnums.drop(cols_drop, 1, inplace=True)
df_colnums.columns = ['date', 'morning', 'night',
                              'food0', 'food1', 'food2', 'food3', 'food4',
                              'food5', 'food6', 'food7', 'food8', 'food9',
                              'alcohol', 'vegetables', 'fruits', 'fats', 'starches']

df_colnums.drop([0, 1], inplace=True)
df_colnums.reset_index(drop=True, inplace=True)
df_colnums.alcohol = df_colnums.alcohol.apply(lambda x: 5 if x == 'multiple' else x)

today = pd.to_datetime(dt.today())
df_input = df_colnums[df_colnums['date'] < today]
df_weight = df_input[df_input.columns[0:3]].copy()
df_text = pd.DataFrame(df_input['date']).copy()
df_diet = pd.DataFrame(df_input['date']).copy()
df_text[df_input.columns[3:13]] = df_input[df_input.columns[3:13]].copy()
df_diet[df_input.columns[13:]] = df_input[df_input.columns[13:]].copy()

def avg_7day(ser):
    avg_7day = np.zeros(ser.shape)
    avg_7day.fill(np.nan)
    for i in xrange(7, ser.shape[0]):
        season = ser[i-7:i].reshape(-1,1)
        avg_7day[i] = np.nan if pd.isnull(season).any() else season.mean()
    return avg_7day

df_weight_7day = pd.DataFrame(df_input['date']).copy()
df_weight_7day['7day_morning'] = avg_7day(df_weight.morning)
df_weight_7day['7day_night'] = avg_7day(df_weight.night)

def df_plot(df, linestyle='solid', marker=False, alpha=1, c=None):
    for i, col in enumerate(df.columns):
        if col != 'date':
            if marker:
                df[col].plot(marker=marker, alpha=alpha, linestyle=linestyle, label=col, c=c[i])
            else:
                df[col].plot(linestyle=linestyle, label=col, c=c[i])

cmap = [None, 'r', 'b']
df_plot(df_weight, linestyle='none', marker='.', alpha=.3, c=cmap)
df_plot(df_weight_7day, c=cmap)
plt.legend()
plt.show()

plt.close('all')
#df_plot(df_diet)
plt.show()
