from load_data import load_data
from clean import clean
from sklearn.metrics.pairwise import cosine_similarity


def name_sim(name1, name2):

    pass


def engr(df):

    df['text_cos_sim'] = df.apply(lambda row: cosine_similarity(row['github_text'], row['meetup_text']), axis=1)
    df['name_sim'] = df.apply(lambda row: name_sim(row['github_name'], row['meetup_name']), axis=1)


if __name__ == '__main__':
    df = load_data()
    df_clean = clean(df)
    df_engr = engr(df_clean)
