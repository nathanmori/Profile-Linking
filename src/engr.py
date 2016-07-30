from load import *
from clean import *
from sklearn.metrics.pairwise import cosine_similarity
from name_tools import match
import sys


def engr(df_clean):

    """CONSIDER MOVING TO THE FIT/TRANSFORM IN model.py"""

    start = start_time('Feature engineering...')

    df_clean['name_sim'] = df_clean.apply(lambda row: match(row['github_name'], row['meetup_name']), axis=1)
    df_clean.drop(['github_name', 'meetup_name'], axis=1, inplace=True)

    end_time(start)

    return df_clean


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
