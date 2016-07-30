# Pseudo-Code

## load.py
- Store list of columns in relevant tables
- Drop irrelevant columns
- Query data with known labels, store to dataframe
- Return dataframe

## clean.py
- Create match labels by converting profile_pics_matched column to 1s and 0s
- Remove extraneous spaces in github_name and meetup_name
- Convert text vectors from string to array of ints, and fill missing data with 0s
- Return clean data

## engr.py
- 

## model.py
- Pipeline model
