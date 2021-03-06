import os
import pandas as pd

def load_df(path):
    if path[-4:] == ".pkl":
        return pd.read_pickle(path)
    
    return pd.read_csv(path)