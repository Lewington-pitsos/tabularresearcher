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

def load_splits(base_df, folds, pipeline):
    indexer = KfoldIndexer(folds, base_df)

    splits = []

    for fold in folds:
        trn_idx, tst_idx = indexer.get_indices(fold)

        trn = base_df.iloc[val_idx]

        assert(len(base_df) == len(trn) + len(val))

        modified_df, _ = pipeline.apply(base_df, trn)
        val = modified_df.iloc[val_idx]
        trn = modified_df.iloc[trn_idx]

        splits.append(trn, val)
    
    return splits

def kfold_generators(make_generator, make_val_generator, indices, n_folds, shuffle=True, val_prop=0.15):
    assert n_folds > 0
    
    if shuffle:
        indices = random.Random(SEED).sample(indices, len(indices))

    if n_folds == 1:
        if val_prop <= 0:
            return [make_generator(indices)], None
            
        val_index = int(len(indices) * val_prop)
        return [make_generator(indices[val_index:])], [make_val_generator(indices[:val_index])]

    folds = np.array_split(indices, n_folds)
    gens = []
    val_gens = []


    for i, fold in enumerate(folds):
        gens.append(make_generator([index for samples in folds[:i] + folds[i+1:] for index in samples]))
        val_gens.append(make_val_generator(fold))

    return gens, val_gens
