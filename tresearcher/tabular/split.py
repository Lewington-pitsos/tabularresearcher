import random

import numpy as np
from tresearcher.tabular.glob import *

class KfoldIndexer():
    def __init__(self, folds, base_df):
        self.folds = folds
        self.base_df = base_df
        self.splits = np.array_split(range(len(base_df)), folds)
    
    def get_indices(self, fold):  
        return [idx for ary in self.splits[:fold] + self.splits[fold+1:] for idx in ary], self.splits[fold]

    def all_indices(self):  
        return [idx for ary in self.splits[:] for idx in ary]

class SplitIterator():
    def __init__(self, base_df, folds, pipeline, x_cols, y_cols):
        self.index = 0
        self.base_df = base_df
        self.folds = folds
        self.pipeline = pipeline
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.indexer = KfoldIndexer(folds, base_df)

    def __next__(self):
        if self.index >= self.folds:
            self.index = 0
            raise StopIteration

        trn_idx, val_idx = self.indexer.get_indices(self.index)
        self.index += 1

        modified_df, y_scale = self.pipeline.apply(self.base_df, trn_idx)
        val = modified_df.iloc[val_idx]
        trn = modified_df.iloc[trn_idx]

        return trn[self.x_cols], trn[self.y_cols], val[self.x_cols], val[self.y_cols], y_scale
    
    def __iter__(self):
        return self