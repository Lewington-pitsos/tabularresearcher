import datetime
import os
import time
import gc
import json

import numpy as np

from tresearcher.researcher.assist import *
from tresearcher.researcher.records import *

def validate_params(params):
    if not params["title"]:
        raise ValueError("paramaters given did not contain a title")
    if not params["notes"]:
        raise ValueError("paramaters given did not contain accompanying notes")

def reduced_params(params):
    return {k: params[k] for k in params.keys() - {'title', 'results', 'notes'}}

def run_experiment(params, experiment_fn, save_path, **kwargs):
    validate_params(params)
    param_hash = get_hash(params)

    print("running experiment {}\n".format(param_hash))

    results = experiment_fn(**reduced_params(params), **kwargs)

    save_experiment(save_path, "{}_{}".format(params["title"], param_hash), parameters=params, results=results.view())

    return results

