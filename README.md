# Profile Linking
## Linking Profiles Across Social Media Sites By User

Author: Nathan Mori <nathanmori@gmail.com>

### Workflow: CRISP-DM
The workflow of the project followed the steps of CRISP-DM, as visualised below.

<img src="./img/CRISP-DM_Process_Diagram.png" alt="CRISP-DM" width="500">

(wikipedia.org)

#### Business Understanding
Talentful seeks to identify ideal candidates to meet client hiring needs. Social media profiles are a great way to gain information on potential recommendations.

The information available on social media varies from site to site. Identifying which profiles belong to the same user across sites allows us to combine the information available on each and to build complete candidate profiles.

An obvious step is to pair profiles that have similar first and last names. With a name-similarity filter, many potential matches remain for each profile. The goal of this project is to develop a model that classifies matches and non-matches among the name-similarity pairs with an accuracy of 85% or higher.

#### Data Understanding
The social media data for this project is from github and meetup user profiles.  The original data is stored in three tables of a postgres database. The first and second contain text vector data from 220,000 github profiles and 500,000 meetup profiles, respectively. The third table contains all of the github-meetup pairs with name similarity - 45,000 potential matches. Among the name-similarity pairs, there are 11,000 distinct github profiles and 15,000 distinct meetup profiles.

For each name-similarity pair, the following information is available:
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

Ultimately, this resulted in using the following features, in descending order of importance:
- Last name similarity
- Cosine similarity between the github and meetup text vectors
- Median distance between locations listed on the github and meetup profiles
- First name similarity

<img src="./img/0469716_feature_importances_AdaBoostClassifier.png" alt="Feature Importances" width="500">

#### Modeling

#### Evaluation

#### Deployment
