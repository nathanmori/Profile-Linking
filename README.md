# Linking Profiles Across Social Media Sites By User

## CRISP-DM Workflow
<img src="./images/CRISP-DM_Process_Diagram.png" alt="CRISP-DM" style="width: 500px;"/>

<img src="./images/CRISP-DM_Process_Diagram.png" alt="CRISP-DM" style="width: 50px;"/>

### Business Understanding
Talentful seeks to identify ideal candidates to meet client hiring needs. Social media profiles are a great way to gain information on potential recommendations.

The information available on social media varies from site to site. Identifying which profiles belong to the same user across sites allows us to combine the information available on each and to build complete candidate profiles.

An obvious step is to pair profiles that have similar first and last names. With a name-similarity filter, many potential matches remain for each profile. The goal of this project is to develop a model that sorts out matches and non-matches among the name-similarity pairs with an accuracy of 85% or higher.

### Data Understanding
The data for this project is from github and meetup user profiles.  The original data is stored in three tables of a postgres database. The first and second contain text vector data from 150,000 github profiles and 500,000 meetup profiles, respectively. The third table contains all of the github-meetup pairs with name similarity - 45,000 potential matches. Among the potential matches, there are 11,000 distinct github profiles and 15,000 distinct meetup profiles.

For each potential match, the following information is available:
- github id
- meetup id
- profile_pics_matched
- github_name
- meetup_name
- min_dist_km
- avg_dist_km
- median_dist_km
- max_dist_km
- github_text
- meetup_text

### Data Preparation
In order to train and test a model, it is necessary to have labelled data. To accomplish this, an assumption is made, that any potential match with the same profile picture on both github and meetup is a positive match. It is also assumed that no individual has more than one profile on either site. So then known negative matches are identified as all potential matches that are not a known positive but include either the github or the meetup from a known positive. A SQL query extracts these observations, resulting in a labeled subset of the potential matches with about 1,100 known positives and 1,600 known negatives.

The labeled dataset is missing some values. 6 observations do not have any of the four distance values, and 388 observations do not have any github_text.

### Modeling

### Evaluation

### Deployment
