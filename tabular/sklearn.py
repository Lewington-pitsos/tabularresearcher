import pandas as pd
from sklearn.linear_model import LinearRegression

from tabular.pipeline import *
from tabular.metrics import *
from tabular.load import *
from tabular.split import *
from researcher.results import *

def linear_reg_experiment(path, folds, pipeline, fold_pipeline, metrics, x_cols, y_cols):
    base_df, base_rescaler = PIPELINES[pipeline].apply(load_df(path), None)
    base_df = base_df.reset_index()

    folds = SplitIterator(base_df, folds, PIPELINES[fold_pipeline], x_cols, y_cols)
    result_tracker = Results()

    for fold, allocation in enumerate(folds):
        print("--------- Starting fold {} ---------".format(fold))
        trn_x, trn_y, val_x, val_y, rescaler = allocation
        rescaler = rescaler.then(base_rescaler)
        
        model = LinearRegression()
        model.fit(trn_x, trn_y)
        preds = rescaler.apply(model.predict(val_x))
        ground_truth = rescaler.apply(val_y)

        for metric_name in metrics:
            metric = METRICS[metric_name]
            score = metric.fn(preds, ground_truth)
            result_tracker.add(fold=fold, name=metric.name, value=score)
    
    return result_tracker
    

    