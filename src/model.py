from load import *
from clean import *
from engr import *


def model(df_engr):

    y = df_engr.pop('profile_pics_matched').values
    X = df_engr.values

    

if __name__ == '__main__':

    df = load()
    df_clean = clean(df)
    df_engr = engr(df_clean)
    mod = model(df_engr)
