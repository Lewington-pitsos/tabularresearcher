import pandas as pd
from sklearn.linear_model import LinearRegression

from tabular.split import *
from researcher.results import *

def linear_reg_experiment(data_name, folds, pipeline, metrics, x_cols, y_cols):
    base_df = pd.read_pickle(data_name)
    splits = SplitIterator(base_df, folds, pipeline, x_cols, y_cols)
    results = Results()

    for fold, trn_x, trn_y, val_x, val_y in enumerate(splits):
        model = LinearRegression()
        model.fit(trn_x, trn_y)
        preds = model.predict(val_x)

        for metric in metrics:
            score = metric.fn(preds, val_y)
            results.add(fold=fold, metric.name, score)
    
    return results
    

    