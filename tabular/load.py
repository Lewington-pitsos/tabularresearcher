import os
import pandas as pd

from tabular.pipeline import PIPELINES

def load_df(params):
    if os._exists(params["data_name"]):
        return pd.read_csv(params["data_name"])
    
    return PIPELINES[params["pipeline"]](params["data_name"])