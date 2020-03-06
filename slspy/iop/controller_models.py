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
    Base for IOP FIR controllers, the controller architecure is borrowed from SLS state-feedback FIR controller
    '''
    def __init__ (self, Ny=0, Nu=0, FIR_horizon=1):
        self._Ny = Ny # dimension of output
        self._Nu = Nu # dimension of control

        self._FIR_horizon = FIR_horizon

        self._X = []  # = [ X[0], X[1], X[2], ... X[FIR_horizon] ]
        self._W = []  # = [ W[0], W[1], W[2], ... W[FIR_horizon] ]
        self._Y = []  # = [ Y[0], Y[1], Y[2], ... Y[FIR_horizon] ]
        self._Z = []  # = [ Z[0], Z[1], Z[2], ... Z[FIR_horizon] ]

        self._delta = []
        self._hat_y = np.zeros([Ny,1])
        self._XI = []

    @staticmethod
    def _convolve(A,B,ub):
        # perform sum_{tau = 0}^{ub-1} A[tau]B[tau]
        if (len(A) == 0) or (len(B) == 0):
            return 0

        conv = 0
        for tau in range(ub):
            if (tau < len(A)) and (tau < len(B)) and (tau >= 0):
                conv += np.dot(A[tau],B[tau])

        return conv

    @staticmethod
    def _FIFO_insert(FIFO_list,element,max_size):
        FIFO_list.insert(0,element)
        # maintain delta length
        while len(FIFO_list) > max_size:
            FIFO_list.pop(-1)

    def initialize (self, delta0=None):
        # empty initialization
        self._delta = []
        # initialize as zero
        self._hat_y = np.zeros([self._Ny,1])

        # X - I
        X_len = len(self._X)
        self._XI = [None] * X_len
        for tau in range(X_len):
            if tau == 0:
                self._XI[tau] = self._X[tau] - np.eye(self._Ny)
            else:
                self._XI[tau] = self._X[tau]
        
        self._total = self._FIR_horizon + 1

    def getControl(self, y):
        # the controller is Y X^{-1}
        self._FIFO_insert(self._delta, y - self._hat_y, self._total)
        u           = self._convolve(A=self._Y,  B=self._delta, ub=self._total )
        self._hat_y = self._convolve(A=self._XI, B=self._delta, ub=self._total )

        return u