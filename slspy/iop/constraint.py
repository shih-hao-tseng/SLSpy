
from .components import IOPConstraint
import cvxpy as cp
import numpy as np
'''
To create a new IOP constraint, inherit the following base function and customize the specified methods.

class IOPConstraint:
    def addConstraints(self, iop, constraints):
        return constraints
'''

class IOPCons_IOP (IOPConstraint):
    '''
    The descrete-time IOP constrains
    '''
    @staticmethod
    def _convolve(A,B,A_len,B_len):
        # perform sum_{t = 0}^{infty} A[t]B[tau-t]
        AB = 0
        AB_len = A_len
        if AB_len > B_len:
            AB_len = B_len

        for t in range(AB_len):
            AB += A[t] * B[B_len - t]

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

        # iop constraints
        for tau in range(iop._FIR_horizon):
            if tau == 0:
                constraints += [
                    iop._X[0] == G[0] * iop._Y[0] + np.eye(Ny)
                ]
                constraints += [
                    iop._Z[0] == iop._Y[0] * G[0] + np.eye(Nu)
                ]
            else:
                constraints += [
                    iop._X[tau] == self._convolve(G, iop._Y, G_len, tau)
                ]
                constraints += [
                    iop._Z[tau] == self._convolve(iop._Y, G, tau, G_len)
                ]

            constraints += [
                iop._W[tau] == self._convolve(G, iop._Z, G_len, tau)
            ]
            constraints += [
                iop._W[tau] == self._convolve(iop._X, G, tau, G_len)
            ]

        return constraints