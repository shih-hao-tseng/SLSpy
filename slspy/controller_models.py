from .core import ControllerModel
import numpy as np
'''
To create a new controller model, inherit the following base function and customize the specified methods.

class ControllerModel:
    def initialize (self):
        # initialize internal state
    def controlConvergence(self, y, **kwargs):
        # getControl without changing the internal state
        return u
    def getControl(self, y, **kwargs):
        return u
'''

class OpenLoopController (ControllerModel):
    '''
    The controller that gives zero control signal.
    '''
    def __init__ (self, Nu=0):
        self.setDimension(Nu)
    
    def setDimension(self,Nu=0):
        # zero control
        self._u = np.zeros([Nu,1])

    def controlConvergence(self, y):
        return self._u.copy()

    def getControl(self, y):
        return self._u.copy()