import pandas as pd
import numpy as np

class Rescaler():
    def __init__(self):
        self.fns = []
        self.finalized = False
    
    def add(self, fn):
        if self.finalized:
            raise ValueError("attempt to add to a finalized rescaler")

        self.fns.append(fn)
    
    def finalize(self):
        self.fns.reverse()
        self.finalized = True

    def apply(self, values):
        if not self.finalized:
            raise ValueError("attempt to apply to an un-finalized rescaler")

        for fn in self.fns:
            values = fn(values)
        return values
    
    def then(self, other):
        """Returns a new Rescaler which applies this Rescaler and then the given Rescaler.
        """
        if not self.finalized:
            raise ValueError("attempt to chain from an un-finalized rescaler")
        if not other.finalized:
            raise ValueError("attempt to chain to an un-finalized rescaler")
            
        chained_rescaler = Rescaler()
        chained_rescaler.add(other.apply)
        chained_rescaler.add(self.apply)
        #NOTE: .finalize() below will reverse the order of application.
        chained_rescaler.finalize()

        return chained_rescaler

class Pipeline():
    def __init__(self, procs, targets, rescaler=None):
        self.procs = procs
        self.targets = targets
        self.rescaler = Rescaler()

    def apply(self, df, trn_idx):
        for proc in self.procs:
            df, scale_fn = proc(df, trn_idx, self.targets)
            if scale_fn:
                self.rescaler.add(scale_fn)
        
        self.rescaler.finalize()
        finalized_rescaler = self.rescaler
        self.rescaler = Rescaler()

        return df, finalized_rescaler


# ---------------------------------------------------------------------------------------
#
#                                        PROCS
#
# ---------------------------------------------------------------------------------------

def target_log1n_proc(df, trn_idx, target):
    df[target] = np.log1p(df[target])
    return df, np.expm1
def reduce_proc(cols):
    return lambda df, trn_idx, targets: (df[cols], None)

def make_datetime_proc(col):
    def datetime_proc(df, idx, target):
        df[col] = pd.to_datetime(df[col])
        return df, None
    
    return datetime_proc

PIPELINES = {
    "noop": Pipeline([], ["meter_reading"])
}