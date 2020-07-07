from sklearn.metrics import *
import numpy as np

from tresearcher.researcher.metric import *

metrics = [
    Metric("mse", mean_squared_error),
    Metric("rmsle", lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, a_min=0, a_max=None)))),
]

METRICS = {m.name:m for m in metrics}