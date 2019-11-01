from .base import ObjBase
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
        return self._u.copy()

class SLS_FIR_Controller (ControllerModel):
    '''
    State base for SLS controllers
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
        pass

    def getControl(self, y):
        # zero control
        return np.zeros([self._Nu,1])

    @staticmethod
    def _convolve(A,B,lower_bound,upper_bound,offset):
        # perform sum_{tau >= lower_bound}^{upper_bound-1} A[tau]B[tau-offset]
        if (len(A) == 0) or (len(B) == 0):
            return np.empty([1,1])

        conv = np.zeros([A[0].shape[0],B[0].shape[1]])

        for tau in range(lower_bound,upper_bound):
            if (tau < len(A)) and (tau-offset < len(B)) and (tau-offset >= 0):
                conv += np.dot(A[tau],B[tau-offset])

        return conv

# we don't combine the state-feedback and non-state-feedback controllers using a switch due to performance
class SLS_State_Feedback_FIR_Controller (SLS_FIR_Controller):
    '''
    State feedback SLS controller with finite impulse response
    '''
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

        u           = self._convolve(A=self._Phi_u, B=self._delta, lower_bound=0, upper_bound=self._FIR_horizon,offset=0)
        self._hat_x = self._convolve(A=self._Phi_x, B=self._delta, lower_bound=1, upper_bound=self._FIR_horizon,offset=1)

        return u

class SLS_Output_Feedback_FIR_Controller (SLS_FIR_Controller):
    '''
    Output feedback SLS controller with finite impulse response
    '''
    def __init__ (self, Nx=0, Nu=0, Ny=0, FIR_horizon=1):
        SLS_FIR_Controller.__init__(Nx=Nx, Nu=Nu, FIR_horizon=FIR_horizon)
        self._Ny = Ny # dimension of measurement