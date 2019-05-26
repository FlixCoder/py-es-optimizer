import numpy as np

"""
TODO:
- add optimizers
- evaluators pass, splited by training and test
- full implementation with the multiple optimizing strategies. best would be parallel
- save and load method
"""


#class definition
class ESOpt:
    """ Evolution-Strategy Optimizer class. Optimizing parameters in a black box fashion. """
    
    #shared attributes (class variables)
    
    #methods
    def __init__(self, params = None):
        #attribute initialization
        self.params = params
        self.steps = 0
    #}
#}
