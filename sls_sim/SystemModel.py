from Base import ObjBase
import numpy as np

class SystemModel (ObjBase):
    '''
    The base class for discrete-time system models.
    A controller synthesizer takes a system model and synthesizes a controller.
    '''
    def __init__ (self):
        self._x = np.empty([0]) # state
        self._y = np.empty([0]) # measurement
        self._z = np.empty([0]) # regularized output

    def systemProgress (self,**kwargs):
        # this function takes the input and progress to next time 
        pass

    def getState(self):
        return self._x

    def getMeasurement(self):
        return self._y
    
    def getOutput(self):
        return self._z

class LTISystem(SystemModel):
    '''
    Contains all matrices of an LTI system as per (3.1)
    Inherits handle class with deep copy functionality
    '''
    def __init__(self):
        # state       : x(t+1)= A*x(t)  + B1*w(t)  + B2*u(t)
        self._A  = np.empty([0,0])
        self._B1 = np.empty([0,0])
        self._B2 = np.empty([0,0])

        # reg output  : z_(t) = C1*x(t) + D11*w(t) + D12*u(t)
        self._C1  = np.empty([0,0])
        self._D11 = np.empty([0,0])
        self._D12 = np.empty([0,0])

        # measurement : y(t)  = C2*x(t) + D21*w(t) + D22*u(t)
        self._C2  = np.empty([0,0])
        self._D21 = np.empty([0,0])
        self._D22 = np.empty([0,0])

        # state dimension and actuator dimension
        self._Nx = 0
        self._Nw = 0
        self._Nu = 0

        self._ignore_output = True
        self._state_feedback = True
    
    def initialize (self, x0, **kwargs):
        # set x0
        self._Nx = x0.shape[0]
        self._x  = x0

    def sanityCheck (self):
        # check the system parameters are coherent
        self._Nx = self._A.shape[0]
        self._Nw = self._B1.shape[1]
        self._Nu = self._B2.shape[1]

        if self._A.shape[1] != self._Nx:
            self.errorMessage('Dimension mismatch: A')
            return False
        
        # TODO
        
        return True
    
    def ignoreOutput (self, ignore_output=True):
        self._ignore_output = ignore_output

    def stateFeedback (self, state_feedback=True):
        self._state_feedback = state_feedback

    def systemProgress(self, u, w, **kwargs):
        self._x =
            np.dot (self._A, self._x) +
            np.dot (self._B1, w) + 
            np.dot (self._B2, u)

        if w.shape[0] != self._Nw:
            self.errorMessage('Dimension mismatch: w')
            return

        if u.shape[0] != self._Nu:
            self.errorMessage('Dimension mismatch: u')
            return

        if not self._ignore_output:
            self._z = 
                np.dot (self._C1, self._x) +
                np.dot (self._D11, w) + 
                np.dot (self._D12, u)

        if not self._state_feedback:
            self._y = 
                np.dot (self._C2, self._x) +
                np.dot (self._D21, w) + 
                np.dot (self._D22, u)