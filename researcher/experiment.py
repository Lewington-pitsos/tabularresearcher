import datetime
import os
import time
import gc
import json

import numpy as np

from researcher.assist import *
from researcher.records import *

def validate_params(params):
    if not params["title"]:
        raise ValueError("paramaters given did not have a title")
    if not params["notes"]:
        raise ValueError("paramaters given did not have accompanying notes")
    
def save_run(params, results, path):
    param_hash = get_hash(params)

    save_experiment(path, "{}_{}".format(params["description"], param_hash), parameters=params, results=results)

def run_experiment(params, experiment_fn, experiment_path, **kwargs):
    validate_params(params)
    results = experiment_fn(**params, **kwargs)
    save_run(params, results.view(), experiment_path)
