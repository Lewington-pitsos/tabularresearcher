from tabular.sklearn import *
import researcher.experiment as exp

EXPERIMENTS = {
    "linear_reg": linear_reg_experiment,
}

def run_experiment(params, save_path, **kwargs):
    experiment_fn = EXPERIMENTS[params["experiment"]]
    exp.run_experiment(params, experiment_fn, save_path, **kwargs)