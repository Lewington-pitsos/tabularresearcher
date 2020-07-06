from collections import defaultdict

class Results():
    """Results provides an api to handle the collection and analysis of experiment results
    """
    def __init__():
        # {fold: metric: [value, value, value, ...]}
        self.__results = defaultdict(lambda: defaultdict(lambda : []))
    
    def add(self, fold, name, value):
        self.results[fold][name].append(value)

    def get_metric(self, target_metric):
        fold_wise = []
        for fold, metrics in self.results:
                fold_wise.append(metrics[target_metric])

        return fold_wise

    def get_agg_metric(self, target_metric, agg_fn):
        fold_wise = []
        for fold, metrics in self.results:
                fold_wise.append(metrics[target_metric])

        return agg_fn(np.array(fold_wise), axis=0)

    def readable_view(self):
        return self.__results.values()
    
    def view(self):
        return self.__results
