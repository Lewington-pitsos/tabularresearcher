from tabular.sklearn import *
from researcher import run 

EXPERIMENTS = {
    "linear_reg": linear_reg_experiment,
}

def run_experiment(params, save_path, **kwargs):
    experiment_fn = EXPERIMENTS[params["experiment"]]
    run.run_experiment(params, experiment_fn, save_path, **kwargs)