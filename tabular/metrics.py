import sklearn

from researcher.metric import *

METRICS = {
    "mse": Metric("mse", sklearn.metrics.mean_squared_error),
}