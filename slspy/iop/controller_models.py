from ..core import ControllerModel
import numpy as np
'''
To create a new controller model, inherit the following base function and customize the specified methods.

class ControllerModel:
    def initialize (self):
        # initialize internal state
    def getControl(self, y, **kwargs):
        return u
'''

class IOP_FIR_Controller (ControllerModel):
    '''
    State base for IOP controllers
    '''
    def __init__ (self, Ny=0, Nu=0, FIR_horizon=1):
        self._Ny = Ny # dimension of output
        self._Nu = Nu # dimension of control

        self._FIR_horizon = FIR_horizon

        self._X = []  # = [ X[0], X[1], X[2], ... X[FIR_horizon-1] ]
        self._W = []  # = [ W[0], W[1], W[2], ... W[FIR_horizon-1] ]
        self._Y = []  # = [ Y[0], Y[1], Y[2], ... Y[FIR_horizon-1] ]
        self._Z = []  # = [ Z[0], Z[1], Z[2], ... Z[FIR_horizon-1] ]

    def initialize (self, delta0=None):
        pass

    def getControl(self, y):
        # zero control
        return np.zeros([self._Nu,1])