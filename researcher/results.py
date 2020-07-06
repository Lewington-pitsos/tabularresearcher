from collections import defaultdict

class Results():
    """Results provides an api to handle the collection and analysis of experiment results
    """
    def __init__():
        # {allocation: fold: metric: [value, value, value, ...]}
        self.__results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda : [])))
    
    def add(self, allocation, fold, name, value):
        self.results[allocation][fold][name].append(value)

    def show_metric(self, metric):
        allocation_wise = []
        for allocation, folds in self.results:
            fold_wise = []                
            for fold, metrics in folds:
                fold_wise.append(metrics[metric])

        return allocation_wise

    def read(self):
        return {"results": self.__results}
