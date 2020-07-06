from collections import defaultdict

class DictRecorder():
    def __init__(self):
        super().__init__()

        results = {}
    
    def write(self, name, value):
        """All values must be written in chronolical order (by name).
        """
        if not self.results[name]:
            self.results[name] = []
        
        self.results[name].append(value)

class FoldRecorder():
    def __init__(self):
        self.folds = defaultdict(lambda: DictRecorder())

    def write(self, fold, name, value):
        """All values must be written in chronolical order (by fold and name).
        """
        self.folds[fold].write(name, value)
    
class Recorder():
    """Capable of tracking all values associated with an experiment.
    """
    def __init__(self):
        self.misc_recorder = DictRecorder()
        self.fold_recorder = FoldRecorder()
    
    def write_fold(self, fold, name, value):
        self.fold_recorder.write(fold, name, value)
    
    def write_misc(self, name, value):
        self.misc_recorder.write(name, value)