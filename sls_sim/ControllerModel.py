from .Base import ObjBase
import numpy as np

class ControllerModel (ObjBase):
    '''
    The base class for discrete-time controller.
    '''
    def initialize (self):
        # initialize internal state
        pass

    def getControl(self, y, **kwargs):
        return None

class OpenLoopController (ControllerModel):
    '''
    The controller that gives zero control signal.
    '''
    def __init__ (self, Nu=0):
        self.setDimension(Nu)
    
    def setDimension(self,Nu=0):
        self._u = np.zeros([Nu,1])

    def getControl(self, y):
        return self._u

class SLS_State_Feedback_FIR_Controller (ControllerModel):
    '''
    State feedback SLS controller with finite impulse response
    '''
    def __init__ (self, Nx=0, Nu=0, FIR_horizon=1):
        self._Nx = Nx # dimension of state
        self._Nu = Nu # dimension of control

        self._FIR_horizon = FIR_horizon

        self._Phi_x = []
        self._Phi_u = []
        self._delta = []
        self._hat_x = np.zeros([Nx,1])

    def initialize (self, delta0=None):
        # empty initialization
        self._delta = []
        if delta0 is not None:
            if isinstance (delta0,list):
                for delta in delta0:
                    self.__addDeltaIfValid(delta)
            else:
                self.__addDeltaIfValid(delta0)

        # initialize as zero
        self._hat_x = np.zeros([self._Nx,1])

    def __addDeltaIfValid(self,delta=None):
        # check if the content is valid
        if ((delta.shape[0] == self._Nx) and
            (delta.shape[1] == 1)):
            self._delta.append(delta)

    def getControl(self, y):
        self._delta.insert(0,y - self._hat_x)
        # maintain delta length
        while len(self._delta) > self._FIR_horizon:
            self._delta.pop(-1)

        u           = self.__convolve(A=self._Phi_u, B=self._delta, lower_bound=0, upper_bound=self._FIR_horizon)
        self._hat_x = self.__convolve(A=self._Phi_x, B=self._delta, lower_bound=1, upper_bound=self._FIR_horizon)

        return u
    
    @staticmethod
    def __convolve(A,B,lower_bound,upper_bound):
        # perform sum_{tau >= lower_bound}^{upper_bound-1} A[tau]B[tau]
        if (len(A) == 0) or (len(B) == 0):
            return np.empty([1,1])

        conv = np.zeros([A[0].shape[0],B[0].shape[1]])

        for tau in range(lower_bound,upper_bound):
            if (tau < len(A)) and (tau < len(B)):
                conv += np.dot(A[tau],B[tau])

        return conv