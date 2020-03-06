from .core import SystemModel, error_message
import numpy as np
'''
To create a new system model, inherit the following base function and customize the specified methods.

class SystemModel:
    def __init__ (self,
        ignore_output=False,
        state_feedback=True
    ):
    def systemProgress (self, u, w=None, **kwargs):
        # this function takes the input and progress to next time 
'''

class LTI_System (SystemModel):
    '''
    Contains all matrices of an LTI system
    '''
    def __init__(self, Nx=0, Nw=0, Nu=0, Ny=0, Nz=0, **kwargs):
        SystemModel.__init__(self, **kwargs)

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
        self._Nw = Nw  # noise
        self._Nu = Nu  # control
        self._Nz = Nz  # output
        self._Ny = Ny  # measurement

    def initialize (self, x0=None):
        SystemModel.initialize(self)

        if x0 is None:
            # use the previous x0
            if self._x0 is None:
                # use zero initialization if self._Nx is set
                if self._Nx > 0:
                    self._x0 = np.zeros([self._Nx,1])
            x0 = self._x0
        else:
            self._x0 = x0

        # set x0
        self._Nx = x0.shape[0]
        self._x  = x0

        # initializing output and measurements by treating w and u to be zeros.
        # one might change this part for some other initialization strategies
        if not self._ignore_output:
            if self._C1 is None:
                self.errorMessage('C1 is not defined when the system output (z) is not ignored. Initialization fails.')
            else:
                self._z = np.dot (self._C1, self._x)

        if not self._state_feedback:
            if self._C2 is None:
                self.errorMessage('C2 is not defined for an output-feedback system. Initialization fails.')
            else:
                self._y = np.dot (self._C2, self._x)

    def sanityCheck (self):
        # check the system parameters are coherent
        if self._Nx == 0:
            return self.errorMessage('Zero dimension (missing initialization): x')

        Nx = self._Nx
        Nw = self._Nw
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
            Nz = self._Nz = (
                self._C1.shape[0]  if self._C1 is not None else
                self._D11.shape[0] if self._D11 is not None else
                self._D12.shape[0] if self._D12 is not None else
                None
            )
            if self._Nz is None:
                return self.errorMessage('None of C1, D11, D12 is set while the output is not ignorable.')

            if self._C1 is None:
                self._C1 = np.zeros([Nz,Nx])
            elif self._C1.shape[1] != Nx:  # remark: no need to check self._C1.shape[0] == Nz
                return self.errorMessage('Dimension mismatch: C1')

            if self._D11 is None:
                self._D11 = np.zeros([Nz,Nw])
            elif ((self._D11.shape[0] != Nz) or
                  (self._D11.shape[1] != Nw)):
                return self.errorMessage('Dimension mismatch: D11')

            if self._D12 is None:
                self._D12 = np.zeros([Nz,Nu])
            elif ((self._D12.shape[0] != Nz) or
                  (self._D12.shape[1] != Nu)):
                return self.errorMessage('Dimension mismatch: D12')
    
        if not self._state_feedback:
            Ny = self._Ny = (
                self._C2.shape[0]  if self._C2 is not None else
                self._D21.shape[0] if self._D21 is not None else
                self._D22.shape[0] if self._D22 is not None else
                None
            )

            if self._C2 is None:
                self._C2 = np.zeros([Ny,Nx])
            elif self._C2.shape[1] != Nx:  # remark: no need to check self._C2.shape[0] == Ny
                return self.errorMessage('Dimension mismatch: C2')

            if self._D21 is None:
                self._D21 = np.zeros([Ny,Nw])
            elif ((self._D21.shape[0] != Ny) or
                  (self._D21.shape[1] != Nw)):
                return self.errorMessage('Dimension mismatch: D21')

            if self._D22 is None:
                self._D22 = np.zeros([Ny,Nu])
            elif ((self._D22.shape[0] != Ny) or
                  (self._D22.shape[1] != Nu)):
                return self.errorMessage('Dimension mismatch: D22')

        return True

    def systemProgress (self, u, w=None):
        if u.shape[0] != self._Nu:
            return self.errorMessage('Dimension mismatch: u')

        if w is not None:
            if w.shape[0] != self._Nw:
                return self.errorMessage('Dimension mismatch: w')

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

    def updateActuation (self,new_act_ids=[]):
        '''
        make a new system with the dynamics of the old system and updated
        actuation (based on rfd output)
        '''
        sys = self.__copy()
        sys._B2  = self._B2 [:, new_act_ids]
        sys._D12 = self._D12[:, new_act_ids]
        sys._Nu  = len(new_act_ids)
        return sys
    
    def __copy(self):
        sys = LTI_System()
        # state
        sys._A  = self._A
        sys._B1 = self._B1

        # reg output
        sys._C1  = self._C1
        sys._D11 = self._D11
        sys._D12 = self._D12

        # measurement
        sys._C2  = self._C2
        sys._D22 = self._D22

        # vector dimensions
        sys._Nx = self._Nx
        sys._Nz = self._Nz
        sys._Ny = self._Ny
        sys._Nw = self._Nw

        return sys


class LTI_FIR_System (SystemModel):
    '''
    The LTI FIR system with
        y = G    u + P_yw w
        z = P_zu u + P_zw w
    '''
    def __init__ (self, Nw=0, Nu=0, Ny=0, Nz=0, **kwargs):
        SystemModel.__init__(self, **kwargs)

        self._Nw = Nw
        self._Nx = self._Ny = Ny
        self._Nz = Nz
        self._Nu = Nu

        self._G = []
        self._Pyw = []
        self._Pzu = []
        self._Pzw = []

        self._u = []
        self._w = []

        self._state_feedback = False

        self._x = self._y = np.zeros([Ny,1])
        self._z = np.zeros([Nz,1])

    @staticmethod
    def _convolve(G,u):
        # perform sum_{t = 0}^{infty} G[t]u[t]
        Gu = 0
        convolution_len = min(len(G),len(u))
        for tau in range(convolution_len):
            Gu += np.dot(G[tau],u[tau])
        return Gu

    def systemProgress (self, u, w=None, **kwargs):
        self._u = [u] + self._u
        self._w = [w] + self._w

        while (len(self._u) > len(self._G)) and (len(self._u) > len(self._Pzu)):
            self._u.pop(-1)
        while (len(self._w) > len(self._Pyw)) and (len(self._w) > len(self._Pzw)):
            self._w.pop(-1)

        self._x = self._y = self._convolve(self._G, self._u) + self._convolve(self._Pyw, self._w)
        self._z = self._convolve(self._Pzu, self._u) + self._convolve(self._Pzw, self._w)

def truncate_LTI_System_to_LTI_FIR_System (system=None,FIR_horizon=1):
    '''
    y = (C2 (zI - A)^{-1} B2 + D22) u + (C2 (zI - A)^{-1} B1 + D21) w
        G = C2 (zI - A)^{-1} B2 + D22
          = D22 + z^{-1} C2 ( 1 - z^{-1} A + z^{-2} A^2 - z^{-3} A^3 ... ) B2
        Pyw = C2 (zI - A)^{-1} B1 + D21
            = D21 + z^{-1} C2 ( 1 - z^{-1} A + z^{-2} A^2 - z^{-3} A^3 ... ) B1

    z = (C1 (zI - A)^{-1} B2 + D12) u + (C1 (zI - A)^{-1} B1 + D11) w
        Pzu = D12 + z^{-1} C1 ( 1 - z^{-1} A + z^{-2} A^2 - z^{-3} A^3 ... ) B2
        Pzw = D11 + z^{-1} C1 ( 1 - z^{-1} A + z^{-2} A^2 - z^{-3} A^3 ... ) B1
    '''
    if not isinstance(system,LTI_System):
        error_message('The system must be LTI_System')
        return

    if system._state_feedback:
        Ny = system._Nx
    else:
        Ny = system._Ny
    Nu = system._Nu
    Nw = system._Nw

    if not system._ignore_output:
        Nz = system._Nz

    truncated_system = LTI_FIR_System (Ny=Ny, Nz=Nz, Nu=Nu, Nw=Nw)

    if FIR_horizon <= 0:
        return truncated_system

    B2 = system._B2
    B1 = system._B1
    mA  = -system._A
    if system._state_feedback:
        C2  = np.eye(Ny)
        D22 = np.zeros([Ny,Nu])
        D21 = np.zeros([Ny,Nw])
    else:
        C2  = system._C2
        D22 = system._D22
        D21 = system._D21
    
    tmp_y = C2

    truncated_system._G = [None] * FIR_horizon
    truncated_system._G[0] = D22

    truncated_system._Pyw = [None] * FIR_horizon
    truncated_system._Pyw[0] = D21

    for t in range(1,FIR_horizon):
        truncated_system._G[t]   = np.dot(tmp_y,B2)
        truncated_system._Pyw[t] = np.dot(tmp_y,B1)
        tmp_y = np.dot(tmp_y,mA)

    if not system._ignore_output:
        tmp_z = 0 if system._C1 is None else system._C1

        truncated_system._Pzu = [None] * FIR_horizon
        truncated_system._Pzu[0] = np.zeros([Nz,Nu]) if system._D12 is None else system._D12 

        truncated_system._Pzw = [None] * FIR_horizon
        truncated_system._Pzw[0] = np.zeros([Nz,Nw]) if system._D11 is None else system._D11

        for t in range(1,FIR_horizon):
            truncated_system._Pzu[t] = np.dot(tmp_z,B2)
            truncated_system._Pzw[t] = np.dot(tmp_z,B1)
            tmp_z = np.dot(tmp_z,mA)

    return truncated_system