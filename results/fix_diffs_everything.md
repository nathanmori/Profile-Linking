n [1]: run model read write
Modeling...              DONE (291.26 seconds).

LogisticRegression
 Test Accuracy: 76.1%
  Filtered Test Accuracy: 86.3%
  Filtered (w/ train) Test Accuracy: 88.4%
RandomForestClassifier
FEATURE IMPORTANCES
name_sim                       1.0
text_sim_tfidf                 0.39437828019
text_norm_diff                 0.295830369273
text_norm_github               0.294512990021
text_sim                       0.2843631969
text_norm_meetup               0.223009147907
min_dist_km                    0.20204094445
avg_dist_km                    0.189645943199
median_dist_km                 0.174387414871
DIFF:avg_dist_km-median_dist_km 0.159389140159
DIFF:median_dist_km-max_dist_km 0.156068155208
DIFF:min_dist_km-median_dist_km 0.151953584614
DIFF:avg_dist_km-max_dist_km   0.15125803671
DIFF:min_dist_km-avg_dist_km   0.140718859189
max_dist_km                    0.132097363078
DIFF:min_dist_km-max_dist_km   0.129789464467
DONE (297.38 seconds).

RandomForestClassifier
 Test Accuracy: 80.9%
  Filtered Test Accuracy: 85.0%
  Filtered (w/ train) Test Accuracy: 86.6%
GradientBoostingClassifier
FEATURE IMPORTANCES
text_norm_github               1.0
text_norm_diff                 0.44623889447
text_sim_tfidf                 0.432506360168
DIFF:avg_dist_km-median_dist_km 0.390130927053
avg_dist_km                    0.369000777536
name_sim                       0.359067850205
min_dist_km                    0.269323950813
text_sim                       0.246383992745
text_norm_meetup               0.234452289877
max_dist_km                    0.233779089513
DIFF:avg_dist_km-max_dist_km   0.168455879389
median_dist_km                 0.164683158937
DIFF:median_dist_km-max_dist_km 0.160077454933
DIFF:min_dist_km-median_dist_km 0.155608475269
DIFF:min_dist_km-avg_dist_km   0.149629551993
DIFF:min_dist_km-max_dist_km   0.0874577295829
DONE (297.89 seconds).

GradientBoostingClassifier
 Test Accuracy: 84.7%
  Filtered Test Accuracy: 87.0%
  Filtered (w/ train) Test Accuracy: 88.8%
AdaBoostClassifier
FEATURE IMPORTANCES
text_norm_github               1.0
text_sim                       0.315789473684
text_sim_tfidf                 0.263157894737
text_norm_diff                 0.210526315789
min_dist_km                    0.105263157895
DIFF:min_dist_km-avg_dist_km   0.105263157895
DIFF:avg_dist_km-median_dist_km 0.105263157895
DIFF:median_dist_km-max_dist_km 0.105263157895
text_norm_meetup               0.105263157895
name_sim                       0.105263157895
avg_dist_km                    0.0526315789474
median_dist_km                 0.0526315789474
max_dist_km                    0.0526315789474
DIFF:avg_dist_km-max_dist_km   0.0526315789474
DIFF:min_dist_km-median_dist_km 0.0
DIFF:min_dist_km-max_dist_km   0.0
DONE (298.15 seconds).

AdaBoostClassifier
 Test Accuracy: 82.6%
  Filtered Test Accuracy: 86.2%
  Filtered (w/ train) Test Accuracy: 88.1%
DONE (298.72 seconds).

SVC
 Test Accuracy: 75.3%
  Filtered Test Accuracy: 87.5%
  Filtered (w/ train) Test Accuracy: 90.0%


Best Model with predict_proba: GradientBoostingClassifier
Test Accuracy: 84.7%
There are 1354 train observations
There are 1354 test observations
There are 546 train matches
There are 554 predicted matches
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 38 duplicates in github_pred_matches
There are 16 duplicates in meetup_pred_matches
There are 66 duplicates in github_train_and_pred_matches
There are 47 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups
github: 613
meetups: [33614 28239  8608]
github: 3498
meetups: [39954 26476]
github: 3759
meetups: [  887 75804]
github: 5871
meetups: [25180  8464]
github: 7654
meetups: [58163 75834]
github: 8253
meetups: [10535 10536]
github: 8899
meetups: [96992 57863 25640]
github: 12604
meetups: [104990  58489]
github: 13108
meetups: [82980 21219]
github: 14235
meetups: [81331  4356]
github: 15547
meetups: [67234  3464]
github: 16591
meetups: [60629 10075]
github: 17416
meetups: [25529 19229]
github: 17763
meetups: [133074  75186 103163]
github: 18362
meetups: [ 10371 125331  84981]
github: 18606
meetups: [1881 1882]
github: 19088
meetups: [ 14041 121327 151737   7950   5435  95068]
github: 19444
meetups: [10077 28836]
github: 19925
meetups: [19229 25529]
github: 20439
meetups: [ 8267 81887]
github: 20918
meetups: [81901 91354]
github: 21097
meetups: [82114 54956 34906]
github: 21147
meetups: [55721 14324]
github: 21643
meetups: [122784 122785]
github: 24622
meetups: [ 55059 121853  22764  90280]
github: 36703
meetups: [68504 79272]
github: 44109
meetups: [ 9855 28743]

meetups linked to multiple githubs
meetup: 2107
githubs: [39737 17622]
meetup: 8320
githubs: [20409 20694]
meetup: 8733
githubs: [20411  7044]
meetup: 10075
githubs: [ 6610 16591 13226]
meetup: 10077
githubs: [15381 19444]
meetup: 18904
githubs: [21041  7599]
meetup: 19229
githubs: [19925 17416]
meetup: 25529
githubs: [17416 19925]
meetup: 33614
githubs: [  613  6197 15434]
meetup: 37909
githubs: [18335  8334]
meetup: 39835
githubs: [ 8528 16071 38860]
meetup: 47745
githubs: [6495 6965]
meetup: 81331
githubs: [14235 15233]
There are 1354 train observations
There are 1354 test observations
There are 546 train matches
There are 454 predicted matches
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 0 duplicates in github_pred_matches
There are 0 duplicates in meetup_pred_matches
There are 3 duplicates in github_train_and_pred_matches
There are 2 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups

meetups linked to multiple githubs

