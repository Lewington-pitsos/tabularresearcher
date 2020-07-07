from tresearcher.researcher import glob
from tresearcher.researcher.results import *

class Experiment():
    def __init__(self, data):
        if data[glob.RESULTS]:
            self.results = Results(data[glob.RESULTS])
        
        self.data = data

    