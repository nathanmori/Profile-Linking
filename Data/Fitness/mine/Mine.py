import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
from datetime import date as dt
import matplotlib.pyplot as plt
import pdb
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import normalize

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
        season = ser[i-7:i].reshape(-1, 1)
        avg_7day[i] = np.nan if pd.isnull(season).any() else season.mean()
    return avg_7day

def avg_7day_cont(ser):
    avg_7day = np.zeros(ser.shape)
    avg_7day.fill(np.nan)
    for i in xrange(7, ser.shape[0]):
        season = ser[i-7:i].reshape(-1, 1)
        avg_7day[i] = ser[i-1] if pd.isnull(season).all() else np.nanmean(season)
    return avg_7day

def diff_7day(ser):
    diff_7day = np.zeros(ser.shape)
    diff_7day.fill(np.nan)
    for i in xrange(7, ser.shape[0]):
        season = ser[[i-7, i]].reshape(-1, 1)
        diff_7day[i] = np.nan if pd.isnull(season).any() else season[1] - season[0]
    return diff_7day

df_weight_7day = pd.DataFrame(df_input['date']).copy()
df_weight_7day['7day_morning'] = avg_7day(df_weight.morning)
df_weight_7day['7day_night'] = avg_7day(df_weight.night)

df_weight_7day_cont = pd.DataFrame(df_input['date']).copy()
df_weight_7day_cont['7day_morning_cont'] = avg_7day_cont(df_weight.morning)
df_weight_7day_cont['7day_night_cont'] = avg_7day_cont(df_weight.night)

#df_weight_diff_7day = pd.DataFrame(df_input['date']).copy()
#df_weight_diff_7day['diff_7day_morning'] = diff_7day(df_weight_7day['7day_morning'])
#df_weight_diff_7day['diff_7day_night'] = diff_7day(df_weight_7day['7day_night'])
df_weight_diff_7day_cont = pd.DataFrame(df_input['date']).copy()
df_weight_diff_7day_cont['diff_7day_morning'] = diff_7day(df_weight_7day_cont['7day_morning_cont'])
df_weight_diff_7day_cont['diff_7day_night'] = diff_7day(df_weight_7day_cont['7day_night_cont'])

df_diet_7day_cont = pd.DataFrame(df_input['date']).copy()
df_diet_7day_cont['7day_alcohol'] = avg_7day_cont(df_diet.alcohol)
df_diet_7day_cont['7day_vegetables'] = avg_7day_cont(df_diet.vegetables)
df_diet_7day_cont['7day_fruits'] = avg_7day_cont(df_diet.fruits)
df_diet_7day_cont['7day_fats'] = avg_7day_cont(df_diet.fats)
df_diet_7day_cont['7day_starches'] = avg_7day_cont(df_diet.starches)

def df_plot(df, linestyle='solid', marker=False, alpha=1, c=None):
    for i, col in enumerate(df.columns):
        if col != 'date':
            if marker:
                if c:
                    df[col].plot(marker=marker, alpha=alpha, linestyle=linestyle, label=col, c=c[i])
                else:
                    df[col].plot(marker=marker, alpha=alpha, linestyle=linestyle, label=col)
            else:
                if c:
                    df[col].plot(linestyle=linestyle, label=col, c=c[i])
                else:
                    df[col].plot(linestyle=linestyle, label=col)

cmap = [None, 'r', 'b']
df_plot(df_weight, linestyle='none', marker='.', alpha=.5, c=cmap)
#df_plot(df_weight_7day, c=cmap)
df_plot(df_weight_7day_cont, linestyle='dashed', c=[None, 'g', 'y'])
plt.legend()
#plt.show()
plt.close('all')

cmap = cmap + ['r', 'c', 'm']
df_plot(df_diet, linestyle='none', marker='o', alpha=.5, c=cmap)
df_plot(df_diet_7day_cont, c=cmap)
plt.legend()
#plt.show()
plt.close('all')


cmap = [None, 'r', 'b']
df_plot(df_weight_diff_7day_cont, linestyle='none', marker='.', alpha=.5, c=cmap)
df_plot(df_weight_diff_7day_cont, c=cmap)
plt.legend()
#plt.show()
plt.close('all')

X_diet_cols = df_diet_7day_cont.columns[1:]
X_weight_cols = df_weight_7day_cont.columns[1]
y_cols = df_weight_diff_7day_cont.columns[1]

X_diet = df_diet_7day_cont[X_diet_cols].values
X_weight = df_weight_7day_cont[X_weight_cols].values
X = np.concatenate([X_diet,
                    X_weight.reshape(-1, 1)],
                    axis=1)
y = df_weight_diff_7day_cont[y_cols].values

X = X[10:-4]
y = y[14:]

Xy = np.concatenate([X, y.reshape(-1,1)], axis=1)
Xy_cols = X_diet_cols.tolist()
Xy_cols.append(X_weight_cols)
Xy_cols.append(y_cols)
df_Xy = pd.DataFrame(Xy, columns=Xy_cols)
scatter_matrix(df_Xy, figsize=(15,9))
plt.show()
plt.close('all')



##STDNORM
X_mean = X.mean(0)
X_std = X.std(0, ddof=1)
X_stdnorm = (X - X_mean)/X_std
X = X_stdnorm


#inverse of weight to get larger effect for lower weight
#X[:, -1] = np.apply_along_axis(lambda x: np.exp(x), axis=0, arr=X[:, -1])


##NEED TO DO STDNORM FOR TRAIN ONLY

#X = add_constant(X)

train = 150
X_train = X[:train]
y_train = y[:train]
X_test = X[train:]
y_test = y[train:]

#lr = LR(n_jobs=-1)
lr = Ridge(alpha=0.1)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
score_train = lr.score(X_train, y_train)
score_test = lr.score(X_test, y_test)
print 'train R^2', score_train
print 'test R^2', score_test

print 'intercept', lr.intercept_
cols = Xy_cols[:-1]
for col, coef in zip(cols, lr.coef_):
    print col, coef


'''
sumsq = 0
for yp, yt in zip(y_pred, y_test):
    print yp, yt
    sumsq += (yp - yt)**2
print 'sumsq', sumsq
print 'rsq', sumsq**.5'''

x_train = range(train)
x_test = range(train, train + len(y_test))
x_all = range(train + len(y_test))

plt.plot(x_train, y_train, label='train_data', marker='o')
plt.plot(x_train, lr.predict(X_train), label='train_pred')
plt.plot(x_test, y_test, label='test_data', marker='o')
plt.plot(x_test, y_pred, label='test_pred')

plt.plot(x_all, X[:, 0], label='alcohol')
plt.plot(x_all, X[:, 1], label='vegetables')
plt.plot(x_all, X[:, 2], label='fruits')
plt.plot(x_all, X[:, 3], label='fats')
plt.plot(x_all, X[:, 4], label='starches')
plt.plot(x_all, X[:, 5], label='weight')
plt.legend()
plt.show()
plt.close('all')




'''
X = sm.add_constant(X)

#####CHECK
#print np.argwhere(np.isnan(y))
#print pd.value_counts([i for i, j in np.argwhere(np.isnan(X))])

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Quantities of interest can be extracted directly from the fitted model. Type ``dir(results)`` for a full list. Here are some examples:
print('Parameters: ', results.params)
print('R2: ', results.rsquared)'''
