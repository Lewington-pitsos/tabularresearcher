import pandas as pd

class Pipeline():
    def __init__(self, procs, targets):
        self.procs = procs
        self.targets = targets

    def apply(self, df, trn_idx):
        for proc in procs:
            df, trn_idx = proc(df, idx, targets)
        
        return df, trn_idx


# ---------------------------------------------------------------------------------------
#
#                                        PROCS
#
# ---------------------------------------------------------------------------------------

def agg_proc(group_cols, agg_col, aggs):
    def aggregated(df, trn_idx, targets):
        if agg_col in targets:
            stats_df = df.iloc[trn_idx]
        else:
            stats_df = df

        grouped = pd.groupby(stats_df, group_cols)
        
        for agg in aggs:
            agg_name = agg_col + "_" + agg
            agg = grouped.agg(agg)
            agg = agg[[group_cols] + [agg_col]]
            agg[agg_col] = agg_name

            merge = pd.merge(df, agg, on=group_cols, suffix=())

            df["_".join(group_cols) + "_wise_" + agg_name] = merge[agg_name] 
        
        return grouped, trn_idx
    
    return aggregated

def count_enc_proc(col):
    return agg_proc([col], [col], ["count"])

PIPELINES = {
    
}