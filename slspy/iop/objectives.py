from .components import IOP_Objective
import numpy as np
import cvxpy as cp
'''
To create a new IOP constraint, inherit the following base function and customize the specified methods.

class IOP_Objective:
    def __init__ (self):
    def getObjectiveValue(self):
        return self._objective_expression
    def addObjectiveValue(self, iop, objective_value):
        return objective_value
'''

class IOP_Obj_H2(IOP_Objective):
    '''
    return                      2
        || P_zw + P_zu Y P_yw ||_H2
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

    def addObjectiveValue(self, iop, objective_value):
        self._objective_expression = 0

        Pyw = iop._system_model._Pyw
        Pzw = iop._system_model._Pzw
        Pzu = iop._system_model._Pzu

        Y = iop._Y

        len_Y   = len(Y)
        len_Pyw = len(Pyw)
        len_Pzu = len(Pzu)
        len_Y_Pyw = len_Y+len_Pyw-1

        Y_Pyw = [self._convolve(Y,len_Y,Pyw,len_Pyw,tau) for tau in range(len_Y_Pyw) ]
        len_Sum = len_Pzu+len_Y_Pyw-1
        Sum = [self._convolve(Pzu,len_Pzu,Y_Pyw,len_Y_Pyw,tau) for tau in range(len_Sum) ]

        len_Pzw = len(Pzw)
        for tau in range(len_Sum):
            if tau < len_Pzw:
                Sum[tau] += Pzw[tau]
            self._objective_expression += cp.sum_squares(Sum[tau])
        
        return objective_value + self._objective_expression

class IOP_Obj_EquivalentH2(IOP_Objective):
    '''
    return                    2
        || [ W,     X - I ] ||
        || [ Z - I, Y     ] ||_H2
    '''
    def addObjectiveValue(self, iop, objective_value):
        self._objective_expression = 0
        X = iop._X
        W = iop._W
        Y = iop._Y
        Z = iop._Z

        Ny = iop._system_model._Ny
        Nu = iop._system_model._Nu

        X_len = len(X)
        if X_len == 0:
            return objective_value

        matrix = [[W[0], X[0] - np.eye(Ny)],[Z[0] - np.eye(Nu), Y[0]]]

        self._objective_expression += cp.sum_squares(
            cp.bmat(matrix)
        )

        for tau in range(1,X_len):
            matrix = [[W[tau], X[tau]],[Z[tau], Y[tau]]]
            self._objective_expression += cp.sum_squares(
                cp.bmat(matrix)
            )

        return objective_value + self._objective_expression
