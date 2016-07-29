from load import *
from clean import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from name_tools import match
import sys


def engr(df_clean):

    start = start_time('Feature engineering...')

    #X_github = np.array(df_clean['github_text'].apply(lambda x: x.flatten().tolist()).tolist())
    #X_meetup = np.array(df_clean['meetup_text'].apply(lambda x: x.flatten().tolist()).tolist())
    #pdb.set_trace()

    #tfidf_github = TfidfTransformer()
    #tfidf_meetup = TfidfTransformer()
    
    #X_tfidf = tfidf.fit_transform(X)

    df_clean['text_sim'] = df_clean.apply(lambda row: float(cosine_similarity(row['github_text'], row['meetup_text'])), axis=1)
    df_clean['name_sim'] = df_clean.apply(lambda row: match(row['github_name'], row['meetup_name']), axis=1)

    df_clean.drop(['github_name', 'meetup_name', 'github_text', 'meetup_text'], axis=1, inplace=True)

    end_time(start)

    return df_clean


if __name__ == '__main__':
    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
