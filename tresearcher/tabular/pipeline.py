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

def agg_proc(group_cols, agg_col, aggs, agg_names=None, fill_missing=False):
    if agg_names:
        if len(agg_names) != len(aggs):
            raise ValueError("If custom names are provided, one name is required for each aggregation function. Received aggregations {}, names {}".format(aggs, agg_names))
    
    def aggregated(df, trn_idx, targets):       
        if agg_col in targets:
            stats_df = df.iloc[trn_idx]
            if not fill_missing:
                for col in group_cols:
                    if stats_df[col].nunique() < df[col].nunique():
                        raise ValueError("{} values, {} for column {} exist in the training dataframe but not the validation dataframe. Ensure that the training dataframe covers all groups".format(df[col].nunique() - stats_df[col].nunique(), set(df[col].unique()) - set(stats_df[col].unique()), col))
                        
        else:
            stats_df = df

        grouped = stats_df[group_cols + [agg_col]].groupby(group_cols)
        
        for i, agg in enumerate(aggs):
            agg_name = agg_col + "_" + agg
            agg = grouped.agg(agg)

            if agg_names:
                new_name = agg_names[i]
            else:
                new_name = "-".join(group_cols) + "_wise_" + agg_name
                
            agg = agg.rename({agg_col: new_name}, axis=1)
            
            df = pd.merge(df, agg, how="left", on=group_cols, suffixes=("", ""))
            
            if fill_missing:
                df = df.fillna(df[new_name].mean())
            
        return df, None
    
    return aggregated

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