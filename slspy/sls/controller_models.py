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

class SLS_FIR_Controller (ControllerModel):
    '''
    State base for SLS controllers
    '''
    def __init__ (self, Nx=0, Nu=0, FIR_horizon=1):
        self._Nx = Nx # dimension of state
        self._Nu = Nu # dimension of control

        self._FIR_horizon = FIR_horizon

        self._Phi_x = []  # = [ 0, Phi_x[1], Phi_x[2], ... Phi_x[FIR_horizon] ]
        self._Phi_u = []  # = [ 0, Phi_u[1], Phi_u[2], ... Phi_u[FIR_horizon] ]
        self._delta = []
        self._hat_x = np.zeros([Nx,1])

    def initialize (self, delta0=None):
        pass

    def getControl(self, y):
        # zero control
        return np.zeros([self._Nu,1])

    @staticmethod
    def _convolve(A,B,lb,ub,offset):
        # perform sum_{tau = lb}^{ub-1} A[tau]B[tau-offset]
        if (len(A) == 0) or (len(B) == 0):
            return 0

        conv = 0
        for tau in range(lb,ub):
            if (tau < len(A)) and (tau-offset < len(B)) and (tau-offset >= 0):
                conv += np.dot(A[tau],B[tau-offset])

        return conv

    @staticmethod
    def _FIFO_insert(FIFO_list,element,max_size):
        FIFO_list.insert(0,element)
        # maintain delta length
        while len(FIFO_list) > max_size:
            FIFO_list.pop(-1)

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
        #TODO: check
        self._FIFO_insert(self._delta, y - self._hat_x, self._FIR_horizon+1)
        u           = self._convolve(A=self._Phi_u, B=self._delta, lb=1, ub=self._FIR_horizon+1, offset=1)
        self._hat_x = self._convolve(A=self._Phi_x, B=self._delta, lb=2, ub=self._FIR_horizon+1, offset=2)

        return u

class SLS_Output_Feedback_FIR_Controller (SLS_FIR_Controller):
    '''
    Output feedback SLS controller with finite impulse response
    '''
    def __init__ (self, Nx=0, Nu=0, Ny=0, D22=None, FIR_horizon=1):
        SLS_FIR_Controller.__init__(self, Nx=Nx, Nu=Nu, FIR_horizon=FIR_horizon)
        self._Ny = Ny # dimension of measurement
        self._D22 = D22

        self._Phi_xx = []  # = [ 0,         Phi_xx[1], Phi_xx[2], ... Phi_xx[FIR_horizon] ]
        self._Phi_ux = []  # = [ 0,         Phi_ux[1], Phi_ux[2], ... Phi_uy[FIR_horizon] ]
        self._Phi_xy = []  # = [ 0,         Phi_xy[1], Phi_xy[2], ... Phi_xy[FIR_horizon] ]
        self._Phi_uy = []  # = [ Phi_uy[0], Phi_uy[1], Phi_uy[2], ... Phi_uy[FIR_horizon] ]

    def initialize (self):
        self._beta = []
        self._bar_y = []
        self.precaculation()

    def getControl(self, y):
        '''
        z beta = tilde_Phi_xx beta + tilde_Phi_xy bar_y
             u = tilde_Phi_ux beta +       Phi_uy bar_y
        where
            tilde_Phi_xx = z (I - z Phi_xx)
            tilde_Phi_ux = z Phi_ux
            tilde_Phi_xy = -z Phi_xy

            bar_y = y - D22 u
        '''

        # decouple u[t] = u' + Phi_uy[0] bar_y[t]
        # we also need to enforce bar_y[t] = y - D_22 u[t]
        # so we have to solve for u[t] and bar_y[t] together
        # derivation:
        #   u[t] = u' + Phi_uy[0] (y[t] - D22 u[t])
        #        = (I + Phi_uy[0] D22)^{-1} (u' + Phi_uy[0] y[t])
        #        = u_multiplier * (u' + Phi_uy[0] y[t])

        u_prime = (
            self._convolve(A=self._tilde_Phi_ux, B=self._beta,  lb=0, ub=self._FIR_horizon,   offset=0) +
            self._convolve(A=self._Phi_uy,       B=self._bar_y, lb=1, ub=self._FIR_horizon+1, offset=1)
        )
        u = np.dot(self._u_multiplier, u_prime + np.dot(self._Phi_uy[0], y))

        self._FIFO_insert(self._bar_y, y - np.dot(self._D22,u), self._FIR_horizon+1)

        z_beta = (self._convolve(A=self._tilde_Phi_xx, B=self._beta,  lb=0, ub=self._FIR_horizon, offset=0) +
                  self._convolve(A=self._tilde_Phi_xy, B=self._bar_y, lb=0, ub=self._FIR_horizon, offset=0))
        self._FIFO_insert(self._beta, z_beta, self._FIR_horizon)

        return u
    
    def precaculation(self):
        # since Phi_xx[1] = I, we have z (I-z Phi_xx) = -Phi_xx[2] - z^{-1} Phi_xx[3] ...
        
        self._tilde_Phi_xx = [] # = [ tilde_Phi_xx[0], tilde_Phi_xx[1], ... ]
        self._tilde_Phi_ux = [] # = [ tilde_Phi_ux[0], tilde_Phi_ux[1], ... ]
        self._tilde_Phi_xy = [] # = [ tilde_Phi_xy[0], tilde_Phi_xy[1], ... ]
        
        for i in range (self._FIR_horizon):
            self._tilde_Phi_ux.append( self._Phi_ux[i+1])
            self._tilde_Phi_xy.append(-self._Phi_xy[i+1])
            if i > 0:
                self._tilde_Phi_xx.append(-self._Phi_xx[i+1])
        
        self._u_multiplier = np.linalg.inv( np.eye(self._D22.shape[1]) + np.dot(self._Phi_uy[0], self._D22) )