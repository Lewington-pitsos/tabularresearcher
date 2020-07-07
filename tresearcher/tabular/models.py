import pandas as pd
from sklearn.linear_model import LinearRegression
import xgboost as xgb


from tresearcher.tabular.pipeline import *
from tresearcher.tabular.metrics import *
from tresearcher.tabular.load import *
from tresearcher.tabular.split import *
from tresearcher.researcher.results import *


def lr_maker(**kwargs):
    return lambda: LinearRegression(**kwargs)
def xgb_maker(**kwargs):
    return lambda: xgb.XGBRegressor(**kwargs)

MODEL_MAKERS = {
    "linear_regression": lr_maker,
    "xgb": xgb_maker,
}
 
 # ---------------------------------------------------------------------------------------------------------
 #
 #                                                  TRAINING LOOPS
 #
 # ---------------------------------------------------------------------------------------------------------

def single_epoch_experiment(model, path, folds, pipeline, fold_pipeline, metrics, x_cols, y_cols, **kwargs):
    base_df, base_rescaler = PIPELINES[pipeline].apply(load_df(path), None)
    model_maker = MODEL_MAKERS[model](**kwargs)

    folds = SplitIterator(base_df, folds, PIPELINES[fold_pipeline], x_cols, y_cols)
    result_tracker = Results()

    for fold, allocation in enumerate(folds):
        print("--------- Starting fold {} ---------".format(fold))
        trn_x, trn_y, val_x, val_y, rescaler = allocation
        rescaler = rescaler.then(base_rescaler)
        
        model = model_maker()
        model.fit(trn_x, trn_y)

        preds_trn = rescaler.apply(model.predict(trn_x))
        ground_truth_trn = rescaler.apply(trn_y)

        preds_val = rescaler.apply(model.predict(val_x))
        ground_truth_val = rescaler.apply(val_y)

        for metric_name in metrics:
            metric = METRICS[metric_name]

            score = metric.fn(ground_truth_trn, preds_trn)
            result_tracker.add(fold=fold, name=metric.name, value=score)

            score = metric.fn(ground_truth_val, preds_val)
            result_tracker.add(fold=fold, name="val_" + metric.name, value=score)
    
    return result_tracker
    

    