from collections import defaultdict

import numpy as np

class Results():
    """Results provides an api to handle the collection and analysis of experiment results
    """
    def __init__(self, results=None):
        # {fold: metric: [value, value, value, ...]}
        self.__results = results or []
    
    def add(self, fold, name, value):
        if len(self.__results) == fold:
            self.__results.append(defaultdict(lambda : []))
        if len(self.__results) < fold:
            raise ValueError("Attempt to write to fold {} when results {} only contains {} folds. It looks like a fold has been skipped".format(fold, self.__results, len(self.__results)))
        if len(self.__results) > fold + 1:
            raise ValueError("Attempt to write to fold {} when results {} contains {} folds already. We shouldn't be writing to already finalized folds.".format(fold, self.__results, len(self.__results)))

        self.__results[fold][name].append(value)

    def get_metric(self, target_metric):
        return [metrics[target_metric] for metrics in self.__results]

    def get_fold_aggregated_metric(self, target_metric, agg_fn):
        fold_wise = []
        for metrics in self.__results:
            fold_wise.append(metrics[target_metric])

        return agg_fn(np.array(fold_wise), axis=0)
    
    def active_fold(self):
        return len(self.__results) - 1
    
    def view(self):
        return self.__results