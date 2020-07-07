import pandas as pd
from sklearn.linear_model import LinearRegression

from tabular.pipeline import *
from tabular.metrics import *
from tabular.load import *
from tabular.split import *
from researcher.results import *

def linear_reg_experiment(data_name, folds, pipeline, metrics, x_cols, y_cols):
    base_df = load_df(data_name)
    splits = SplitIterator(base_df, folds, PIPELINES[pipeline], x_cols, y_cols)
    results = Results()

    for fold, allocation in enumerate(splits):
        trn_x, trn_y, val_x, val_y = allocation
        
        model = LinearRegression()
        model.fit(trn_x, trn_y)
        preds = model.predict(val_x)

        for metric_name in metrics:
            metric = METRICS[metric_name]
            score = metric.fn(preds, val_y)
            results.add(fold=fold, name=metric.name, value=score)
    
    return results
    

    