n [1]: run model read
Modeling...
# Train Obs: 1354
# Test Obs:  1354
fit pipeline
fit text
fit missing
trans missing
False    1151
True      203
Name: github_text_missing, dtype: int64
False    1354
Name: meetup_text_missing, dtype: int64
False    1151
True      203
Name: text_missing, dtype: int64
trans text
trans missing
False    1151
True      203
Name: github_text_missing, dtype: int64
False    1354
Name: meetup_text_missing, dtype: int64
False    1151
True      203
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
trans pipeline
trans text
trans missing
False    1151
True      203
Name: github_text_missing, dtype: int64
False    1354
Name: meetup_text_missing, dtype: int64
False    1151
True      203
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
trans text
trans missing
False    1169
True      185
Name: github_text_missing, dtype: int64
False    1354
Name: meetup_text_missing, dtype: int64
False    1169
True      185
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
DONE (223.39 seconds).

LogisticRegression
 Test Accuracy: 88.8%
  Filtered Test Accuracy: 90.7%
  Filtered (w/ train) Test Accuracy: 93.2%
RandomForestClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
name_sim                       0.396276761018
fullname_similarity            0.330288058575
text_sim_tfidf                 0.153249991959
text_dot_tfidf                 0.125245644625
text_sim                       0.114995500375
min_dist_km                    0.111727767998
DIFF:text_norm                 0.106011236096
text_norm_github               0.103861407658
text_dot                       0.0998619192863
avg_dist_km                    0.0964095052706
median_dist_km                 0.0914216342978
text_norm_meetup               0.0884146975354
DIFF:median_dist_km-max_dist_km 0.0781345740154
max_dist_km                    0.0767750032578
DIFF:min_dist_km-avg_dist_km   0.0766543779916
DIFF:avg_dist_km-max_dist_km   0.0766138403073
DIFF:min_dist_km-median_dist_km 0.0724669523248
DIFF:min_dist_km-max_dist_km   0.0709896813031
DIFF:avg_dist_km-median_dist_km 0.0671922836324
DIFF:text_norm_tfidf           0.00655224257298
text_norm_meetup_tfidf         0.00639848451515
text_missing                   0.00616785444662
github_text_missing            0.00612337165341
text_norm_github_tfidf         0.00517444961291
firstname_similarity           0.0041139600231
meetup_text_missing            0.0
DONE (225.47 seconds).

RandomForestClassifier
 Test Accuracy: 88.3%
  Filtered Test Accuracy: 88.4%
  Filtered (w/ train) Test Accuracy: 90.1%
GradientBoostingClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
DIFF:avg_dist_km-median_dist_km 0.895882712423
DIFF:text_norm                 0.800687495584
text_sim                       0.630938656475
text_norm_meetup               0.536844208897
text_norm_github               0.521855569816
text_sim_tfidf                 0.470797728357
min_dist_km                    0.451333393498
text_dot_tfidf                 0.432063550456
avg_dist_km                    0.389318053354
text_dot                       0.384703758089
DIFF:median_dist_km-max_dist_km 0.381678442174
DIFF:avg_dist_km-max_dist_km   0.380345742729
median_dist_km                 0.351394455091
DIFF:min_dist_km-avg_dist_km   0.308126779136
max_dist_km                    0.260985355514
DIFF:min_dist_km-median_dist_km 0.257959827137
DIFF:min_dist_km-max_dist_km   0.231624993679
fullname_similarity            0.173288942679
name_sim                       0.0899488806005
text_norm_meetup_tfidf         0.0458728805218
github_text_missing            0.0263962693779
text_missing                   0.0196626200275
DIFF:text_norm_tfidf           0.0193784452092
text_norm_github_tfidf         0.0150139384925
firstname_similarity           0.0112634530391
meetup_text_missing            0.0
DONE (226.06 seconds).

GradientBoostingClassifier
 Test Accuracy: 89.1%
  Filtered Test Accuracy: 89.9%
  Filtered (w/ train) Test Accuracy: 91.7%
AdaBoostClassifier
FEATURE IMPORTANCES
text_sim                       1.0
DIFF:avg_dist_km-median_dist_km 0.714285714286
text_norm_github               0.571428571429
text_norm_meetup               0.571428571429
text_dot                       0.571428571429
min_dist_km                    0.428571428571
text_dot_tfidf                 0.428571428571
avg_dist_km                    0.285714285714
lastname_similarity            0.285714285714
DIFF:min_dist_km-median_dist_km 0.285714285714
DIFF:avg_dist_km-max_dist_km   0.285714285714
DIFF:median_dist_km-max_dist_km 0.285714285714
text_sim_tfidf                 0.285714285714
DIFF:text_norm                 0.285714285714
name_sim                       0.285714285714
median_dist_km                 0.142857142857
max_dist_km                    0.142857142857
fullname_similarity            0.142857142857
firstname_similarity           0.142857142857
DIFF:min_dist_km-avg_dist_km   0.0
DIFF:min_dist_km-max_dist_km   0.0
github_text_missing            0.0
meetup_text_missing            0.0
text_missing                   0.0
text_norm_github_tfidf         0.0
text_norm_meetup_tfidf         0.0
DIFF:text_norm_tfidf           0.0
DONE (226.34 seconds).

AdaBoostClassifier
 Test Accuracy: 89.4%
  Filtered Test Accuracy: 90.0%
  Filtered (w/ train) Test Accuracy: 92.0%
DONE (226.78 seconds).

SVC
 Test Accuracy: 88.7%
  Filtered Test Accuracy: 89.6%
  Filtered (w/ train) Test Accuracy: 91.9%
DONE (226.85 seconds).

XGBClassifier
 Test Accuracy: 89.4%
  Filtered Test Accuracy: 90.8%
  Filtered (w/ train) Test Accuracy: 92.5%


Best Model with predict_proba: AdaBoostClassifier
Test Accuracy: 89.4%

XGB Test Accuracy: 89.4%
  Filtered Test Accuracy: 90.8%
  Filtered (w/ train) Test Accuracy: 92.5%
