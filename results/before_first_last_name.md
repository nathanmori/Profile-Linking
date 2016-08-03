Modeling...
# Train Obs: 1354
# Test Obs:  1354
DONE (246.47 seconds).

LogisticRegression
 Test Accuracy: 78.1%
  Filtered Test Accuracy: 86.3%
  Filtered (w/ train) Test Accuracy: 88.6%
RandomForestClassifier
FEATURE IMPORTANCES
name_sim                       1.0
text_dot_tfidf                 0.314271821497
text_sim_tfidf                 0.30134040249
DIFF:text_norm                 0.292439697823
text_norm_github               0.256848594486
text_sim                       0.237155306494
text_norm_meetup               0.222621424913
min_dist_km                    0.213772708733
avg_dist_km                    0.210445097965
text_dot                       0.196456541738
median_dist_km                 0.19485004119
DIFF:avg_dist_km-median_dist_km 0.1702675946
DIFF:median_dist_km-max_dist_km 0.165407780856
DIFF:avg_dist_km-max_dist_km   0.161169025625
DIFF:min_dist_km-avg_dist_km   0.158214109465
DIFF:min_dist_km-median_dist_km 0.157560467687
max_dist_km                    0.149890223001
DIFF:min_dist_km-max_dist_km   0.141857811719
DIFF:text_norm_tfidf           0.0180735755401
text_norm_github_tfidf         0.0166528768463
text_norm_meetup_tfidf         0.000955391087197
DONE (248.52 seconds).

RandomForestClassifier
 Test Accuracy: 81.1%
  Filtered Test Accuracy: 84.6%
  Filtered (w/ train) Test Accuracy: 86.3%
GradientBoostingClassifier
FEATURE IMPORTANCES
text_norm_github               1.0
DIFF:text_norm                 0.515066295054
name_sim                       0.415026947723
avg_dist_km                    0.388632519695
DIFF:avg_dist_km-median_dist_km 0.377029359722
text_sim                       0.32496471602
min_dist_km                    0.277983186628
text_sim_tfidf                 0.265849229635
max_dist_km                    0.255673820942
DIFF:median_dist_km-max_dist_km 0.23830047596
text_dot_tfidf                 0.194323206565
text_norm_meetup               0.183503187135
DIFF:avg_dist_km-max_dist_km   0.175478159177
median_dist_km                 0.167381209634
text_dot                       0.143239570296
DIFF:min_dist_km-avg_dist_km   0.140552265942
DIFF:min_dist_km-median_dist_km 0.119080685521
DIFF:min_dist_km-max_dist_km   0.058389678024
text_norm_github_tfidf         0.00474639933648
DIFF:text_norm_tfidf           0.00146374264479
text_norm_meetup_tfidf         0.0
DONE (249.08 seconds).

GradientBoostingClassifier
 Test Accuracy: 84.4%
  Filtered Test Accuracy: 86.7%
  Filtered (w/ train) Test Accuracy: 88.3%
AdaBoostClassifier
FEATURE IMPORTANCES
text_norm_github               1.0
text_sim                       0.352941176471
DIFF:avg_dist_km-median_dist_km 0.176470588235
text_norm_meetup               0.176470588235
text_dot_tfidf                 0.176470588235
min_dist_km                    0.117647058824
DIFF:min_dist_km-avg_dist_km   0.117647058824
DIFF:median_dist_km-max_dist_km 0.117647058824
text_sim_tfidf                 0.117647058824
DIFF:text_norm                 0.117647058824
text_dot                       0.117647058824
name_sim                       0.117647058824
median_dist_km                 0.0588235294118
DIFF:min_dist_km-median_dist_km 0.0588235294118
DIFF:min_dist_km-max_dist_km   0.0588235294118
DIFF:avg_dist_km-max_dist_km   0.0588235294118
avg_dist_km                    0.0
max_dist_km                    0.0
text_norm_github_tfidf         0.0
text_norm_meetup_tfidf         0.0
DIFF:text_norm_tfidf           0.0
DONE (249.35 seconds).

AdaBoostClassifier
 Test Accuracy: 82.8%
  Filtered Test Accuracy: 86.7%
  Filtered (w/ train) Test Accuracy: 88.5%
DONE (250.02 seconds).

SVC
 Test Accuracy: 75.8%
  Filtered Test Accuracy: 86.4%
  Filtered (w/ train) Test Accuracy: 88.9%


Best Model with predict_proba: GradientBoostingClassifier
Test Accuracy: 84.4%
There are 1354 train observations
There are 1354 test observations
There are 546 train matches
There are 552 predicted matches
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 43 duplicates in github_pred_matches
There are 15 duplicates in meetup_pred_matches
There are 72 duplicates in github_train_and_pred_matches
There are 44 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups
github: 613
meetups: [33614  8608]
github: 3759
meetups: [  887 75804]
github: 5871
meetups: [25180  8464]
github: 7654
meetups: [ 58163  75834 100285]
github: 8253
meetups: [10535 10536]
github: 8899
meetups: [96992 20402 57863 25640]
github: 12604
meetups: [ 95542 104990  68704  58489  36914]
github: 13108
meetups: [82980 21219]
github: 14235
meetups: [81331  4356]
github: 15547
meetups: [67234  3464  1316]
github: 16591
meetups: [60629 10075]
github: 17416
meetups: [25529 19229]
github: 17763
meetups: [133074  75186 103163]
github: 18362
meetups: [10371 10374 84981]
github: 18606
meetups: [1881 1882]
github: 19088
meetups: [ 14041 121327 151737  10790   7950   5435  95068]
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
githubs: [ 8528 16071]
meetup: 47745
githubs: [6495 6965]
meetup: 81331
githubs: [14235 15233]
There are 1354 train observations
There are 1354 test observations
There are 546 train matches
There are 449 predicted matches
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 0 duplicates in github_pred_matches
There are 0 duplicates in meetup_pred_matches
There are 3 duplicates in github_train_and_pred_matches
There are 2 duplicates in meetup_train_and_pred_matches

githubs linked to multiple meetups

meetups linked to multiple githubs

