import random

import numpy as np
from tabular.glob import *

class KfoldIndexer():
    def __init__(self, folds, base_df):
        self.folds = folds
        self.base_df = base_df
        self.splits = np.array_split(range(len(base_df)), folds)
    
    def get_indices(self, fold):  
        return [idx for ary in self.splits[:fold] + self.splits[fold+1:] for idx in ary], self.splits[fold]

    def all_indices(self):  
        return [idx for ary in self.splits[:] for idx in ary]

def load_splits(base_df, folds, pipeline, x_cols, y_cols):
    indexer = KfoldIndexer(folds, base_df)

    splits = []

    for fold in folds:
        trn_idx, val_idx = indexer.get_indices(fold)

        modified_df, _ = pipeline.apply(base_df, trn_idx)
        val = modified_df.iloc[val_idx]
        trn = modified_df.iloc[trn_idx]

        splits.append(trn[x_cols], trn[y_cols], val[x_cols], val[y_cols])
    
    return splits
