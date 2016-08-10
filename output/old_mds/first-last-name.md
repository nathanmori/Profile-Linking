
deling...
# Train Obs: 1354
# Test Obs:  1354
DONE (245.52 seconds).

LogisticRegression
 Test Accuracy: 88.8%
  Filtered Test Accuracy: 91.1%
  Filtered (w/ train) Test Accuracy: 93.6%
RandomForestClassifier
FEATURE IMPORTANCES
lastname_similarity            1.0
name_sim                       0.389351096778
fullname_similarity            0.351756358647
text_dot_tfidf                 0.156815836005
text_sim_tfidf                 0.135050540229
text_sim                       0.134049467747
DIFF:text_norm                 0.125799020532
text_norm_meetup               0.109484475207
min_dist_km                    0.10847112182
text_norm_github               0.0994624177499
text_dot                       0.096739618315
avg_dist_km                    0.0920440590847
median_dist_km                 0.0919198738698
DIFF:avg_dist_km-max_dist_km   0.0847121658957
DIFF:median_dist_km-max_dist_km 0.0813822456392
max_dist_km                    0.0807177381884
DIFF:avg_dist_km-median_dist_km 0.0736799855262
DIFF:min_dist_km-avg_dist_km   0.0733411469333
DIFF:min_dist_km-max_dist_km   0.0720617243857
DIFF:min_dist_km-median_dist_km 0.069792416321
DIFF:text_norm_tfidf           0.0132577350578
text_norm_github_tfidf         0.00881359830352
firstname_similarity           0.00570477098807
text_norm_meetup_tfidf         0.00190420312288
DONE (247.57 seconds).

RandomForestClassifier
 Test Accuracy: 88.6%
  Filtered Test Accuracy: 88.9%
  Filtered (w/ train) Test Accuracy: 90.8%
GradientBoostingClassifier
FEATURE IMPORTANCES
DIFF:avg_dist_km-median_dist_km 1.0
DIFF:text_norm                 0.999617674414
lastname_similarity            0.92935568941
text_sim                       0.63390552631
text_norm_meetup               0.597225765131
DIFF:avg_dist_km-max_dist_km   0.521720774017
text_norm_github               0.502659106459
text_sim_tfidf                 0.491024136802
min_dist_km                    0.430531178166
avg_dist_km                    0.400113909182
text_dot_tfidf                 0.327051762464
DIFF:median_dist_km-max_dist_km 0.325746411207
text_dot                       0.316314312775
DIFF:min_dist_km-avg_dist_km   0.282446326635
max_dist_km                    0.272891551411
DIFF:min_dist_km-max_dist_km   0.244108057644
DIFF:min_dist_km-median_dist_km 0.21067779551
median_dist_km                 0.192085724322
fullname_similarity            0.166194603262
name_sim                       0.0741526502896
text_norm_meetup_tfidf         0.0242261446877
DIFF:text_norm_tfidf           0.0214640878119
text_norm_github_tfidf         0.00873092445936
firstname_similarity           0.00732006409082
DONE (248.14 seconds).

GradientBoostingClassifier
 Test Accuracy: 89.7%
  Filtered Test Accuracy: 90.5%
  Filtered (w/ train) Test Accuracy: 92.2%
AdaBoostClassifier
FEATURE IMPORTANCES
text_norm_meetup               1.0
DIFF:avg_dist_km-median_dist_km 0.714285714286
text_norm_github               0.571428571429
text_dot                       0.571428571429
min_dist_km                    0.428571428571
fullname_similarity            0.428571428571
text_sim                       0.428571428571
DIFF:text_norm                 0.428571428571
text_dot_tfidf                 0.428571428571
DIFF:min_dist_km-median_dist_km 0.285714285714
DIFF:avg_dist_km-max_dist_km   0.285714285714
DIFF:median_dist_km-max_dist_km 0.285714285714
name_sim                       0.285714285714
avg_dist_km                    0.142857142857
median_dist_km                 0.142857142857
max_dist_km                    0.142857142857
firstname_similarity           0.142857142857
lastname_similarity            0.142857142857
DIFF:min_dist_km-avg_dist_km   0.142857142857
text_sim_tfidf                 0.142857142857
DIFF:min_dist_km-max_dist_km   0.0
text_norm_github_tfidf         0.0
text_norm_meetup_tfidf         0.0
DIFF:text_norm_tfidf           0.0
DONE (248.41 seconds).

AdaBoostClassifier
 Test Accuracy: 88.9%
  Filtered Test Accuracy: 89.9%
  Filtered (w/ train) Test Accuracy: 91.7%
DONE (248.83 seconds).

SVC
 Test Accuracy: 88.6%
  Filtered Test Accuracy: 89.7%
  Filtered (w/ train) Test Accuracy: 92.0%


Best Model with predict_proba: GradientBoostingClassifier
Test Accuracy: 89.7%
There are 1354 train observations
There are 1354 test observations
There are 548 train matches
There are 570 predicted matches
There are 4 duplicates in github_train_matches
There are 3 duplicates in meetup_train_matches
There are 22 duplicates in github_pred_matches
There are 15 duplicates in meetup_pred_matches
There are 48 duplicates in github_train_and_pred_matches
There are 48 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups
github: 5871
meetups: [25180  8464]
github: 9304
meetups: [ 1715 84007]
github: 13108
meetups: [27584 82980 21219]
github: 15233
meetups: [ 4356 81331]
github: 15547
meetups: [67234  1316  3464]
github: 16591
meetups: [60629 10075]
github: 17416
meetups: [25529 19229]
github: 17763
meetups: [133074  75186 103163]
github: 18606
meetups: [1881 1882]
github: 19444
meetups: [10077 28836]
github: 19925
meetups: [19229 25529]
github: 20439
meetups: [ 8267 81887]
github: 20864
meetups: [ 9966 48568]
github: 20918
meetups: [81901 91354]
github: 21007
meetups: [149085   6696]
github: 21097
meetups: [82114 54956]
github: 21147
meetups: [55721 14324]
github: 21643
meetups: [122784 122785]
github: 36703
meetups: [68504 79272]

meetups linked to multiple githubs
meetup: 2107
githubs: [17622 39737]
meetup: 8320
githubs: [20409 20694]
meetup: 8733
githubs: [20411  7044]
meetup: 10075
githubs: [16591 13226  6610]
meetup: 10077
githubs: [15381 19444]
meetup: 18904
githubs: [21041  7599]
meetup: 19107
githubs: [29662  8239]
meetup: 19229
githubs: [19925 17416]
meetup: 25529
githubs: [17416 19925]
meetup: 37909
githubs: [18335  8334]
meetup: 47745
githubs: [6495 6965]
meetup: 81331
githubs: [14235 15233]
meetup: 84007
githubs: [ 9304 12879 21933]
There are 1354 train observations
There are 1354 test observations
There are 548 train matches
There are 487 predicted matches
There are 4 duplicates in github_train_matches
There are 3 duplicates in meetup_train_matches
There are 0 duplicates in github_pred_matches
There are 0 duplicates in meetup_pred_matches
There are 4 duplicates in github_train_and_pred_matches
There are 3 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups

meetups linked to multiple githubs
