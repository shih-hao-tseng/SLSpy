from .components import IOP_Constraint
import cvxpy as cp
import numpy as np
'''
To create a new IOP constraint, inherit the following base function and customize the specified methods.

class IOP_Constraint:
    def addConstraints(self, iop, constraints):
        return constraints
'''

class IOP_Cons_IOP (IOP_Constraint):
    '''
    The descrete-time IOP constrains
    '''
    @staticmethod
    def _convolve(A,A_len,B,B_len,tau):
        # perform sum_{t = 0}^{infty} A[t]B[tau-t]
        AB = 0
        for t in range(A_len):
            tmp = tau - t
            if tmp >= B_len:
                continue
            if tmp < 0:
                continue
            AB += A[t] @ B[tmp]
        return AB

    def addConstraints(self, iop, constraints=[]):
        '''
        IOP constraints:
        [ I, -G ][ X W ] = [ I 0 ]
                 [ Y Z ]
        [ X W ][ -G ] = [ 0 ]
        [ Y Z ][  I ]   [ I ]
        '''
        Ny = iop._system_model._Ny
        Nu = iop._system_model._Nu

        G = iop._system_model._G
        G_len = len(G)
        total = iop._FIR_horizon + 1

        # also the outside convolutions
        Zx = np.zeros([Ny,Ny])
        Zw = np.zeros([Ny,Nu])
        Zz = np.zeros([Nu,Nu])

        # iop constraints
        for tau in range(total+G_len-1):
            if tau < total:
                if tau == 0:
                    constraints += [
                        iop._X[0] == G[0] @ iop._Y[0] + np.eye(Ny)
                    ]
                    constraints += [
                        iop._Z[0] == iop._Y[0] @ G[0] + np.eye(Nu)
                    ]
                else:
                    constraints += [
                        iop._X[tau] == self._convolve(G, G_len, iop._Y, total, tau)
                    ]
                    constraints += [
                        iop._Z[tau] == self._convolve(iop._Y, total, G, G_len, tau)
                    ]

                constraints += [
                    iop._W[tau] == self._convolve(G, G_len, iop._Z, total, tau)
                ]
                constraints += [
                    iop._W[tau] == self._convolve(iop._X, total, G, G_len, tau)
                ]
            else:
                constraints += [
                    Zx == self._convolve(G, G_len, iop._Y, total, tau)
                ]
                constraints += [
                    Zz == self._convolve(iop._Y, total, G, G_len, tau)
                ]
                constraints += [
                    Zw == self._convolve(G, G_len, iop._Z, total, tau)
                ]
                constraints += [
                    Zw == self._convolve(iop._X, total, G, G_len, tau)
                ]

        return constraints

class IOP_Cons_Sparse (IOP_Constraint):
    def __init__(self, S=None):
        # the mask
        self._S = S

    def addConstraints(self, iop, constraints=[]):
        if self._S is None:
            return constraints

        for tau in range(iop._FIR_horizon + 1):
            for ix,iy in np.ndindex(self._S.shape):
                if self._S[ix,iy] == 0:
                    constraints += [ iop._Y[tau][ix,iy] == 0 ]

        return constraints