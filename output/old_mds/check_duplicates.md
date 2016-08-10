Modeling...              DONE (207.66 seconds).
LogisticRegression
  Test Recall:            81.8%
  Train Precision:        68.3%
  # Features:             21
  Train Accuracy:         77.0%
  Test AUC:               88.0%
  Test Accuracy:          78.6%
  Train Recall:           80.0%
  Test Precision:         71.2%
DONE (209.69 seconds).
RandomForestClassifier
  Test Recall:            73.6%
  Train Precision:        99.8%
  # Features:             21
  OOB Accuracy:           77.3%
  Train Accuracy:         99.6%
  Test AUC:               85.4%
  Test Accuracy:          77.9%
  # Useless Features:     3
  Train Recall:           99.1%
  Test Precision:         73.5%
DONE (210.18 seconds).
GradientBoostingClassifier
  Test Recall:            79.5%
  Train Precision:        89.9%
  # Features:             21
  Train Accuracy:         93.7%
  Test AUC:               87.1%
  Test Accuracy:          79.2%
  # Useless Features:     3
  Train Recall:           95.1%
  Test Precision:         73.0%
DONE (210.44 seconds).
AdaBoostClassifier
  Test Recall:            80.7%
  Train Precision:        74.4%
  # Features:             21
  Train Accuracy:         81.8%
  Test AUC:               87.6%
  Test Accuracy:          78.6%
  # Useless Features:     8
  Train Recall:           83.7%
  Test Precision:         71.6%
DONE (210.44 seconds).
GaussianNB
  Test Recall:            83.7%
  Train Precision:        61.7%
  # Features:             21
  Train Accuracy:         72.2%
  Test AUC:               82.8%
  Test Accuracy:          72.5%
  Train Recall:           81.9%
  Test Precision:         62.8%
DONE (211.06 seconds).
SVC
  Test Recall:            91.2%
  Train Precision:        64.8%
  # Features:             21
  Train Accuracy:         76.8%
  Test AUC:               85.9%
  Test Accuracy:          75.6%
  Train Recall:           92.9%
  Test Precision:         64.7%
DONE (211.19 seconds).
LinearSVC
  Test Recall:            92.0%
  Train Precision:        64.2%
  # Features:             21
  Train Accuracy:         76.3%
  Test Accuracy:          75.4%
  Train Recall:           93.2%
  Test Precision:         64.4%
DONE (211.48 seconds).
NuSVC
  Test Recall:            91.2%
  Train Precision:        64.8%
  # Features:             21
  Train Accuracy:         76.7%
  Test Accuracy:          75.6%
  Train Recall:           92.7%
  Test Precision:         64.7%


Best Model with predict_proba: GradientBoostingClassifier
Test Accuracy: 79.2%
probas: [ 0.01350861  0.02796965  0.03353701 ...,  0.01890028  0.31705761
  0.05493468]
There are 3 duplicates in github_train_matches
There are 2 duplicates in meetup_train_matches
There are 110 duplicates in github_pred_matches
There are 21 duplicates in meetup_pred_matches
There are 134 duplicates in github_train_and_pred_matches
There are 54 duplicates in meetup_train_and_pred_matches
