from .Base import ObjBase
from .NoiseModel import NoiseModel, GuassianNoise
import numpy as np

class SystemModel (ObjBase):
    '''
    The base class for discrete-time system models.
    A controller synthesizer takes a system model and synthesizes a controller.
    '''
    def __init__ (self):
        self._x = np.empty([0])  # state
        self._y = np.empty([0])  # measurement
        self._z = np.empty([0])  # regularized output

        self._ignore_output = False
        self._state_feedback = True

        self._noise_model = None

    def initialize (self, x0):
        # set the initial state
        pass

    def systemProgress (self, **kwargs):
        # this function takes the input and progress to next time 
        pass

    def getState(self):
        return self._x

    def getMeasurement(self):
        if self._state_feedback:
            return self._x
        else:
            return self._y
    
    def getOutput(self):
        if self._ignore_output:
            return None
        else:
            return self._z

    def ignoreOutput (self, ignore_output=False):
        self._ignore_output = ignore_output

    def stateFeedback (self, state_feedback=True):
        self._state_feedback = state_feedback

    def useNoiseModel (self, noise_model=None):
        if isinstance (noise_model,NoiseModel):
            self._noise_model = noise_model

class LTISystem(SystemModel):
    '''
    Contains all matrices of an LTI system as per (3.1)
    '''
    def __init__(self,Nx=0,Nw=0,Nu=0,Ny=0,Nz=0):
        SystemModel.__init__(self)

        if not isinstance(Nx,int):
            Nx = 0
        if not isinstance(Nw,int):
            Nw = 0
        if not isinstance(Nu,int):
            Nu = 0
        if not isinstance(Ny,int):
            Ny = 0
        if not isinstance(Nz,int):
            Nz = 0

        # state       : x(t+1)= A*x(t)  + B1*w(t)  + B2*u(t)
        self._A  = None # np.zeros([Nx,Nx])
        self._B1 = None # np.zeros([Nx,Nw])
        self._B2 = None # np.zeros([Nx,Nu])

        # reg output  : z_(t) = C1*x(t) + D11*w(t) + D12*u(t)
        self._C1  = None # np.zeros([Nz,Nx])
        self._D11 = None # np.zeros([Nz,Nw])
        self._D12 = None # np.zeros([Nz,Nu])

        # measurement : y(t)  = C2*x(t) + D21*w(t) + D22*u(t)
        self._C2  = None # np.zeros([Ny,Nx])
        self._D21 = None # np.zeros([Ny,Nw])
        self._D22 = None # np.zeros([Ny,Nu])

        # vector dimensions of
        self._Nx = Nx  # state
        self._Nu = Nu  # control
        self._Nz = Nz  # output
        self._Ny = Ny  # measurement

        self._noise_model = GuassianNoise (Nw=Nw)
    
    def initialize (self, x0=None):
        if x0 is None:
            # use the previous x0
            x0 = self._x0
        else:
            self._x0 = x0
        # set x0
        self._Nx = x0.shape[0]
        self._x  = x0

        # initializing output and measurements by treating w and u to be zeros.
        # one might change this part for some other initialization strategies
        if not self._ignore_output:
            self._z = np.dot (self._C1, self._x)
        if not self._state_feedback:
            self._y = np.dot (self._C2, self._x)

    def sanityCheck (self):
        # check the system parameters are coherent
        if self._Nx == 0:
            return self.errorMessage('Zero dimension (missing initialization): x')

        Nx = self._Nx
        Nw = self._noise_model._Nw
        Nu = self._Nu = self._B2.shape[1]

        # fill in zero matrices if undefined
        if self._A is None:
            self._A = np.zeros([Nx,Nx])
        elif ((self._A.shape[0] != Nx) or
            (self._A.shape[1] != Nx)):
            return self.errorMessage('Dimension mismatch: A')
        
        if self._B1 is None:
            self._B1 = np.zeros([Nx,Nw])
        elif ((self._B1.shape[0] != Nx) or
            (self._B1.shape[1] != Nw)):
            return self.errorMessage('Dimension mismatch: B1')

        if self._B2 is None:
            self._B2 = np.zeros([Nx,Nu])
        elif self._B2.shape[0] != Nx:
            return self.errorMessage('Dimension mismatch: B2')

        if not self._ignore_output:
            self._Nz = self._C1.shape[0]

            if self._C1.shape[1] != self._Nx:
                return self.errorMessage('Dimension mismatch: C1')
            if ((self._D11.shape[0] != self._Nz) or
                (self._D11.shape[1] != Nw)):
                return self.errorMessage('Dimension mismatch: D11')
            if ((self._D12.shape[0] != self._Nz) or
                (self._D12.shape[1] != self._Nu)):
                return self.errorMessage('Dimension mismatch: D12')
    
        if not self._state_feedback:
            self._Ny = self._C2.shape[0]
            if self._C2.shape[1] != self._Nx:
                return self.errorMessage('Dimension mismatch: C2')
            if ((self._D21.shape[0] != self._Ny) or
                (self._D21.shape[1] != Nw)):
                return self.errorMessage('Dimension mismatch: D21')
            if ((self._D22.shape[0] != self._Ny) or
                (self._D22.shape[1] != self._Nu)):
                return self.errorMessage('Dimension mismatch: D22')

        return True

    def systemProgress(self, u):
        if u.shape[0] != self._Nu:
            return self.errorMessage('Dimension mismatch: u')

        if self._noise_model is not None:
            w = self._noise_model.getNoise()

            if not isinstance(w, np.ndarray):
                # in case w is a list
                w = np.array(w)
            
            self._x = (
                np.dot (self._A, self._x) +
                np.dot (self._B1, w) + 
                np.dot (self._B2, u)
            )

            if not self._ignore_output:
                self._z = (
                    np.dot (self._C1, self._x) +
                    np.dot (self._D11, w) + 
                    np.dot (self._D12, u)
                )

            if not self._state_feedback:
                self._y = (
                    np.dot (self._C2, self._x) +
                    np.dot (self._D21, w) + 
                    np.dot (self._D22, u)
                )
        else:
            # noise free
            self._x = (
                np.dot (self._A, self._x) +
                np.dot (self._B2, u)
            )

            if not self._ignore_output:
                self._z = (
                    np.dot (self._C1, self._x) +
                    np.dot (self._D12, u)
                )

            if not self._state_feedback:
                self._y = (
                    np.dot (self._C2, self._x) +
                    np.dot (self._D22, u)
                )