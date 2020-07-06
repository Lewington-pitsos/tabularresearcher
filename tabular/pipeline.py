class Pipeline():
    def __init__(self, procs):
        self.procs = procs

    def apply(self, df, trn_idx):
        for proc in procs:
            df, trn_idx = proc(df, idx)
        
        return df, trn_idx


PIPELINES = {
    
}