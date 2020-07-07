from tresearcher.tabular.models import *
from tresearcher.researcher import run 

EXPERIMENTS = {
    "linear_regression": single_epoch_experiment,
    "xgb": single_epoch_experiment,
}

def run_experiment(params, save_path, **kwargs):
    experiment_fn = EXPERIMENTS[params["model"]]
    return run.run_experiment(params, experiment_fn, save_path, **kwargs)