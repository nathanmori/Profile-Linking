from load import *
from clean import *
from sklearn.metrics.pairwise import cosine_similarity
from name_tools import match
import sys


def engr(df_clean):

    sys.stdout.write('Feature engineering...')
    start_time = time()

    df_clean['text_sim'] = df_clean.apply(lambda row: float(cosine_similarity(row['github_text'], row['meetup_text'])), axis=1)
    df_clean['name_sim'] = df_clean.apply(lambda row: match(row['github_name'], row['meetup_name']), axis=1)

    df_clean.drop(['github_name', 'meetup_name', 'github_text', 'meetup_text'], axis=1, inplace=True)

    end_time(start_time)

    return df_clean


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
