from .core import Controller_Model
import numpy as np
'''
To create a new controller model, inherit the following base function and customize the specified methods.

class Controller_Model:
    def initialize (self):
        # initialize internal state
    def getControl(self, y, **kwargs):
        return u
'''

class Open_Loop_Controller (Controller_Model):
    '''
    The controller that gives zero control signal.
    '''
    def __init__ (self, Nu=0):
        self.setDimension(Nu)
    
    def setDimension(self,Nu=0):
        self._u = np.zeros([Nu,1])

    def getControl(self, y):
        return self._u.copy()