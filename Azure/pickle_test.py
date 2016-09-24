import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


def azureml_main(dataframe1 = None, dataframe2 = None):

    pipe = Pipeline([('step1', TfidfTransformer())])
    pkl = pickle.dumps(pipe, protocol=1)
    df = pd.DataFrame([[pkl]])

    return df



import pickle

def azureml_main(dataframe1 = None, dataframe2 = None):

    pkl = dataframe1.ix[0, 0]
    pipe = pickle.loads(pkl)

    return dataframe1,
