import pandas as pd
from sklearn.linear_model import LinearRegression

from tabular.split import *
from researcher.results import *

def linear_reg_experiment(data_name, folds, pipeline, metrics):
    base_df = pd.read_csv(data_name)
    splits = load_splits(base_df, folds, pipeline)
    results = Results()

    for split, trnX, trny, valX, valy in enumerate(splits):
        model = LinearRegression()

        model.fit(trnX, trny)
        preds = model.predict(valX)
        for metric in metrics:
            score = metric.fn(preds, valy)
            results.add(split, 0, metric.name, score)
    
    return results
    

    