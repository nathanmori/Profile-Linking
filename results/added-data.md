n [1]: run model read
Modeling...
# Train Obs: 2086
# Test Obs:  2087
fit pipeline
fit text
fit missing
trans missing
False    1772
True      314
Name: github_text_missing, dtype: int64
False    2029
True       57
Name: meetup_text_missing, dtype: int64
False    1741
True      345
Name: text_missing, dtype: int64
trans text
trans missing
False    1772
True      314
Name: github_text_missing, dtype: int64
False    2029
True       57
Name: meetup_text_missing, dtype: int64
False    1741
True      345
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
trans pipeline
trans text
trans missing
False    1772
True      314
Name: github_text_missing, dtype: int64
False    2029
True       57
Name: meetup_text_missing, dtype: int64
False    1741
True      345
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
trans text
trans missing
False    1782
True      305
Name: github_text_missing, dtype: int64
False    2031
True       56
Name: meetup_text_missing, dtype: int64
False    1744
True      343
Name: text_missing, dtype: int64
/home/ubuntu/anaconda2/lib/python2.7/site-packages/sklearn/utils/validation.py:420: DataConversionWarning: Data with input dtype object was converted to float64 by StandardScaler.
  warnings.warn(msg, DataConversionWarning)
DONE (336.87 seconds).

LogisticRegression
 Test Accuracy: 89.5%
  Filtered Test Accuracy: 90.8%
  Filtered (w/ train) Test Accuracy: 93.3%
RandomForestClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
name_sim                       0.190140047095
fullname_similarity            0.174882837141
text_sim_tfidf                 0.137998703569
text_dot_tfidf                 0.123074054035
text_sim                       0.0945157675084
text_dot                       0.0851128547224
median_dist_km                 0.0751488899366
min_dist_km                    0.0739745370824
avg_dist_km                    0.0717527853318
text_norm_github               0.0711965356151
text_norm_meetup               0.0701935980819
DIFF:avg_dist_km-max_dist_km   0.0665195398073
DIFF:text_norm                 0.066509224844
DIFF:min_dist_km-median_dist_km 0.0664005582144
DIFF:min_dist_km-avg_dist_km   0.0645561010787
max_dist_km                    0.0632194615159
DIFF:median_dist_km-max_dist_km 0.0615100200434
DIFF:avg_dist_km-median_dist_km 0.0610129754484
DIFF:min_dist_km-max_dist_km   0.0562577566034
meetup_text_missing            0.0453676430489
firstname_similarity           0.032226656296
github_text_missing            0.00751562528664
text_norm_meetup_tfidf         0.00472914666882
DIFF:text_norm_tfidf           0.00426346091141
text_missing                   0.00322910398855
text_norm_github_tfidf         0.0030486374862
DONE (339.03 seconds).

RandomForestClassifier
 Test Accuracy: 88.9%
  Filtered Test Accuracy: 90.2%
  Filtered (w/ train) Test Accuracy: 92.2%
GradientBoostingClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
DIFF:text_norm                 0.744648748734
text_sim                       0.685591720283
DIFF:avg_dist_km-median_dist_km 0.649139253559
text_norm_meetup               0.544730261639
fullname_similarity            0.492620335231
DIFF:min_dist_km-avg_dist_km   0.457646463176
min_dist_km                    0.423614547529
text_dot                       0.39449572001
text_dot_tfidf                 0.370387312865
text_norm_github               0.357178972562
text_sim_tfidf                 0.327165663465
DIFF:median_dist_km-max_dist_km 0.282197679475
median_dist_km                 0.274494882733
avg_dist_km                    0.253244627443
DIFF:min_dist_km-median_dist_km 0.230794736921
DIFF:avg_dist_km-max_dist_km   0.226723444127
DIFF:min_dist_km-max_dist_km   0.178691378626
max_dist_km                    0.129200719332
meetup_text_missing            0.118620641736
firstname_similarity           0.114110455039
name_sim                       0.104519579205
github_text_missing            0.0665697334023
text_missing                   0.0309103398809
text_norm_meetup_tfidf         0.0269154398196
DIFF:text_norm_tfidf           0.0266548833222
text_norm_github_tfidf         0.0130860494579
DONE (339.88 seconds).

GradientBoostingClassifier
 Test Accuracy: 89.3%
  Filtered Test Accuracy: 90.3%
  Filtered (w/ train) Test Accuracy: 92.4%
AdaBoostClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
text_dot_tfidf                 0.75
text_sim                       0.625
fullname_similarity            0.5
text_norm_github               0.5
text_dot                       0.5
min_dist_km                    0.375
median_dist_km                 0.25
DIFF:min_dist_km-median_dist_km 0.25
DIFF:avg_dist_km-median_dist_km 0.25
DIFF:median_dist_km-max_dist_km 0.25
text_norm_meetup               0.25
DIFF:text_norm                 0.25
avg_dist_km                    0.125
max_dist_km                    0.125
firstname_similarity           0.125
meetup_text_missing            0.125
DIFF:min_dist_km-avg_dist_km   0.0
DIFF:min_dist_km-max_dist_km   0.0
DIFF:avg_dist_km-max_dist_km   0.0
github_text_missing            0.0
text_missing                   0.0
text_sim_tfidf                 0.0
text_norm_github_tfidf         0.0
text_norm_meetup_tfidf         0.0
DIFF:text_norm_tfidf           0.0
name_sim                       0.0
DONE (340.24 seconds).

AdaBoostClassifier
 Test Accuracy: 89.4%
  Filtered Test Accuracy: 90.7%
  Filtered (w/ train) Test Accuracy: 92.6%
DONE (341.38 seconds).

SVC
 Test Accuracy: 89.0%
  Filtered Test Accuracy: 91.1%
  Filtered (w/ train) Test Accuracy: 93.7%
DONE (341.46 seconds).

XGBClassifier
 Test Accuracy: 90.2%
  Filtered Test Accuracy: 91.3%
  Filtered (w/ train) Test Accuracy: 93.2%

