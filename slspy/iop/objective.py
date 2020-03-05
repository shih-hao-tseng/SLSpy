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
    return 
        || [ W,     X - I ] ||
        || [ Z - I, Y     ] ||_H2^2
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