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
            AB += A[t] * B[tmp]
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
        horizon = iop._FIR_horizon

        # also the outside convolutions
        Zx = np.zeros([Ny,Ny])
        Zw = np.zeros([Ny,Nu])
        Zz = np.zeros([Nu,Nu])

        # iop constraints
        for tau in range(horizon+G_len-1):
            if tau < horizon:
                if tau == 0:
                    constraints += [
                        iop._X[0] == G[0] * iop._Y[0] + np.eye(Ny)
                    ]
                    constraints += [
                        iop._Z[0] == iop._Y[0] * G[0] + np.eye(Nu)
                    ]
                else:
                    constraints += [
                        iop._X[tau] == self._convolve(G, G_len, iop._Y, horizon, tau)
                    ]
                    constraints += [
                        iop._Z[tau] == self._convolve(iop._Y, horizon, G, G_len, tau)
                    ]

                constraints += [
                    iop._W[tau] == self._convolve(G, G_len, iop._Z, horizon, tau)
                ]
                constraints += [
                    iop._W[tau] == self._convolve(iop._X, horizon, G, G_len, tau)
                ]
            else:
                constraints += [
                    Zx == self._convolve(G, G_len, iop._Y, horizon, tau)
                ]
                constraints += [
                    Zz == self._convolve(iop._Y, horizon, G, G_len, tau)
                ]
                constraints += [
                    Zw == self._convolve(G, G_len, iop._Z, horizon, tau)
                ]
                constraints += [
                    Zw == self._convolve(iop._X, horizon, G, G_len, tau)
                ]

        return constraints