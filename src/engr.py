from load import *
from clean import *
from sklearn.metrics.pairwise import cosine_similarity


def name_sim(name1, name2):

    pass


def engr(df):

    print 'Feature engineering...'
    start_time = time()

    df['text_cos_sim'] = df.apply(lambda row: float(cosine_similarity(row['github_text'], row['meetup_text'])), axis=1)
    df['name_sim'] = df.apply(lambda row: float(name_sim(row['github_name'], row['meetup_name'])), axis=1)

    end_time(start_time)

    return df


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
