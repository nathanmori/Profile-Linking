# Profile Linking
## Linking Profiles Across Social Media Sites By User

Author: [Nathan Mori](https://www.linkedin.com/in/nathanmori)

Email: <nathanmori@gmail.com>

### Tools

The following tools were used on this project.

- [Amazon Elastic Cloud Compute](https://pages.awscloud.com/clicktochat.html)
- [PostgreSQL](https://www.postgresql.org/)
- [Python](https://www.python.org/)
    - [NumPy](http://www.numpy.org/)
    - [SciPy](https://www.scipy.org/)
    - [pandas](http://pandas.pydata.org/)
    - [scikit-learn](http://scikit-learn.org/)
    - [XGBoost](https://github.com/dmlc/xgboost)
    - [Psycopg](http://initd.org/psycopg/)
    - [matplotlib](http://matplotlib.org/)
    - [name_tools](https://pypi.python.org/pypi/name_tools)

### Workflow: CRISP-DM
The workflow for this project followed the steps of CRISP-DM, as visualized below (wikipedia.org).

<img src="./img/CRISP-DM_Process_Diagram.png" alt="CRISP-DM" width="500">

#### Business Understanding
This project was completed on behalf of [Talentful](http://www.talentful.io). Talentful is a talent-seeking company whose mission is to identify ideal candidates to meet the hiring needs of clients. Part of their strategy is to utilize data available on social media sites.

<img src="./img/Talentful.png" alt="CRISP-DM" width="200">

The candidates and information available on social media varies from site to site. One of the challenges is that many people share the same name, and there is no direct connection indicating which profile on one site belongs to the same person as a profile on another site. Linking the profiles by user is a critical step to consolidate many tables into a single dataset that has exactly one entry per person.

An obvious step is to pair profiles that have similar first and last names. With a name-similarity filter, many potential matches remain for each profile. The goal of this project is to develop a model that classifies matches and non-matches among the name-similarity pairs with an accuracy of 85% or higher.

#### Data Understanding
*Note: The data for this project is proprietary property of Talentful and is excluded from this repository due to confidentiality requirements.*

The social media data used for this project is from github and meetup user profiles. The raw data is stored in three tables of a PostgreSQL database. The first and second contain text vector data from 220,000 github profiles and 500,000 meetup profiles, respectively. The third table contains all of the github-meetup pairs with name similarity - 45,000 potential matches. Among the name-similarity pairs, there are 11,000 distinct github profiles and 15,000 distinct meetup profiles.

For each name-similarity pair, the PostgreSQL database contains the following columns:
- profile_pics_matched
- github_name
- meetup_name
- min_dist_km
- avg_dist_km
- median_dist_km
- max_dist_km
- github_text
- meetup_text

#### Data Preparation
In order to train and test a model, it is necessary to have labeled data. To accomplish this, an assumption is made that any potential match with the same profile picture on both github and meetup is a positive match. It is also assumed that no individual has more than one profile on either site. So then known negative matches are identified as all potential matches that are not a known positive but include either the github or the meetup from a known positive. A SQL query extracts these observations, resulting in a labeled subset of the potential matches with 1,766 known positives and 2,407 known negatives. This is about a 40/60 split, which is pretty good class balance.

The labeled dataset is missing some values. 6 observations do not have any of the four distance values, and 388 observations do not have any github_text. Missing distances were simply filled with the median of the train data. Missing text vectors are a different animal. Multiple approaches were evaluated, including filling with zero vectors, filling with the average vector in the train set, or ignoring the vectors until the aggregate features were calculated (cosine similarity, euclidean distance, etc.) and filling with the min, mean, median, or max at that point.

Similarly, there was more than one option for how to evaluate name-similarity. Specifically, the similarity of the full names could be used. What ended up being the better option was to compute two similarity scores - one for the first names, and one for the last names.

There was also a high degree of correlation between the four distances. To avoid multicollinearity in the model, it was desirable to drop all but one. So there was another decision as to which distance should be kept.

To handle the many options for how to use each column of the raw data, all of the feature engineering was developed in scikit-learn style classes with the alternatives as parameters. These could then all be combined in a Pipeline, and GridSearchCV used to find the best features.

Ultimately, this resulted in using the following four features, in descending order of importance:
- Last name similarity
- Cosine similarity between the github and meetup text vectors
- Median distance between locations listed on the github and meetup profiles
- First name similarity

<img src="./output/output_final/0680621_feature_importances_AdaBoostClassifier.png" alt="Feature Importances" width="500">

The resulting feature and target data is shown in the scatter matrix below.

<img src="./output/output_final/0680621_scatter-matrix.png" alt="Scatter Matrix" width="1000">

#### Modeling

The data was split 50/50 into train and test sets. Six classification algorithms were fit to the train data:

- scikit-learn: LogisticRegression
- scikit-learn: SVC
- scikit-learn: RandomForestClassifier
- scikit-learn: GradientBoostingClassifier
- scikit-learn: AdaBoostClassifier
- eXtreme Gradient Boosting: XGBClassifier

There is one problem with the predictions from these algorithms. That is that each name-similarity pair is classified independently of the others. The result is that some profiles are matched with more than one on the other site. This violates the initial assumption that users have at most one profile per site.

So, the predictions were filtered to address the issue of duplicate matches. The approach consisted of sorting the predicted positives in order of descending probability. Then the list was looped over, adding each github and meetup used in a positive pair to a list of "taken" profiles. If another pair re-used a "taken" github or meetup, the prediction for that pair was changed to negative.

That gives us "Test Filtered" results. In other words, all of the duplicates within the test set have been filtered. This does not, however, prevent against profiles from the test set appearing in predicted positives. So, the same algorithm was rerun on the original predictions, except starting with "taken" lists that included all githubs and meetups from known positives in the train set. This produces "Train + Test Filtered" results.

Filtering both the train and the test duplicates is the best approach, and will be used for final predictions. The "Test Filtered" results are included to evaluate the performance of the model. The better "Train + Test Filtered" scores are not scalable. At this point, the ratio of the train and test sample sizes is 1:1. However, when the model is actually used to make predictions, it will be trained on the 4,000 known pairs and be classifying 41,000 unknown pairs. So, the ratio is about 1:10. In other words, the train data cannot share duplicates with as many of the test observations. So the focus is on the more scalable "Test Filtered" metrics.

#### Evaluation

The Unfiltered, Test Filtered, and Train + Test Filtered accuracy scores are listed in the table below. The six algorithms all had comparable performance. The Random Forest was the worst, and AdaBoost was slightly better than the rest.

| Algorithm | Unfiltered | Test Filtered | Train + Test Filtered |
|---|---|---|---|
| LogisticRegression | 89.5% | 90.7% | 93.1% |
| SVC | 89.3% | 90.1% | 92.6% |
| RandomForestClassifier | 85.5% | 87.0% | 89.7% |
| GradientBoostingClassifier | 88.1% | 88.8% | 91.3% |
| **AdaBoostClassifier** | **89.6%** | **91.0%** | **93.3%** |
| XGBClassifier | 89.3% | 90.8% | 93.1% |

In addition to the accuracies, the following table includes the Precision, Recall, and AUC of the AdaBoost algorithm. Talentful's original goal was an accuracy of 85%, so it is great to see the model exceeding expectations and getting 91%.

| AdaBoost Performance | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
|Unfiltered| *89.6%* | 84.8% | 92.6% | 0.931 |
|**Test Filtered**| ***91.0%*** | **89.6%** | **89.6%** | **0.930** |
|Train + Test Filtered| *93.3%* | 96.6% | 87.6% | 0.934 |

The ROC plot from the AdaBoost predictions is shown below. It can be seen that model is performing very well. It also shows that the filtering applied reduces the false positive rate as expected, without drastically decreasing the true positive rate.

<img src="./output/output_final/0680621_ROCs.png" alt="ROC" width="500">

#### Deployment

The model's predictive power can now be deployed to classify the 41,000 unknown name-similarity pairs.
