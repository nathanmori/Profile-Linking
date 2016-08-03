In [1]: run model read write evals
/home/ubuntu/anaconda2/lib/python2.7/site-packages/matplotlib/axes/_base.py:2787: UserWarning: Attempting to set identical left==right results
in singular transformations; automatically expanding.
left=0.0, right=0.0
  'left=%s, right=%s') % (left, right))
/home/ubuntu/anaconda2/lib/python2.7/site-packages/matplotlib/axes/_base.py:3045: UserWarning: Attempting to set identical bottom==top results
in singular transformations; automatically expanding.
bottom=0.0, top=0.0
  'bottom=%s, top=%s') % (bottom, top))
Modeling...              DONE (329.05 seconds).

LogisticRegression
  Test Recall:            80.9%
  Train Precision:        67.0%
  # Features:             22
  Train Accuracy:         76.4%
  Test AUC:               86.7%
  Test Accuracy:          76.0%
  Train Recall:           81.7%
  Test Precision:         67.8%
  Filtered Test Accuracy: 86.3%
  Filtered (w/ train) Test Accuracy: 88.4%
DONE (335.16 seconds).

RandomForestClassifier
  Test Recall:            78.1%
  Train Precision:        99.8%
  # Features:             22
  OOB Accuracy:           79.5%
  Train Accuracy:         99.9%
  Test AUC:               88.4%
  Test Accuracy:          81.6%
  # Useless Features:     3
  Train Recall:           100.0%
  Test Precision:         77.9%
  Filtered Test Accuracy: 85.7%
  Filtered (w/ train) Test Accuracy: 87.0%
DONE (335.72 seconds).

GradientBoostingClassifier
  Test Recall:            80.7%
  Train Precision:        94.8%
  # Features:             22
  Train Accuracy:         97.0%
  Test AUC:               91.5%
  Test Accuracy:          84.8%
  # Useless Features:     3
  Train Recall:           97.8%
  Test Precision:         82.5%
  Filtered Test Accuracy: 86.9%
  Filtered (w/ train) Test Accuracy: 88.7%
DONE (336.00 seconds).

AdaBoostClassifier
  Test Recall:            78.6%
  Train Precision:        81.3%
  # Features:             22
  Train Accuracy:         85.3%
  Test AUC:               89.9%
  Test Accuracy:          82.6%
  # Useless Features:     8
  Train Recall:           82.6%
  Test Precision:         79.6%
  Filtered Test Accuracy: 86.2%
  Filtered (w/ train) Test Accuracy: 88.1%
DONE (336.01 seconds).

GaussianNB
  Test Recall:            46.0%
  Train Precision:        73.3%
  # Features:             22
  Train Accuracy:         71.0%
  Test AUC:               83.2%
  Test Accuracy:          72.3%
  Train Recall:           44.3%
  Test Precision:         78.8%
  Filtered Test Accuracy: 73.9%
  Filtered (w/ train) Test Accuracy: 75.2%
DONE (336.66 seconds).

SVC
  Test Recall:            90.8%
  Train Precision:        64.7%
  # Features:             22
  Train Accuracy:         76.8%
  Test AUC:               85.4%
  Test Accuracy:          75.3%
  Train Recall:           93.4%
  Test Precision:         64.4%
  Filtered Test Accuracy: 88.8%
  Filtered (w/ train) Test Accuracy: 91.3%
DONE (336.83 seconds).

LinearSVC
  Test Recall:            91.9%
  Train Precision:        64.1%
  # Features:             22
  Train Accuracy:         76.2%
  Test Accuracy:          75.4%
  Train Recall:           93.0%
  Test Precision:         64.4%
DONE (337.13 seconds).

NuSVC
  Test Recall:            91.7%
  Train Precision:        65.1%
  # Features:             22
  Train Accuracy:         77.3%
  Test Accuracy:          75.5%
  Train Recall:           94.1%
  Test Precision:         64.5%


Best Model with predict_proba: GradientBoostingClassifier
Test Accuracy: 84.8%
There are 1354 train observations
There are 1354 test observations
There are 546 train matches
There are 553 predicted matches
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 37 duplicates in github_pred_matches
There are 16 duplicates in meetup_pred_matches
There are 65 duplicates in github_train_and_pred_matches
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
meetups: [55059 22764 90280]
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

