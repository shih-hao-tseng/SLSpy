import numpy as np
import cvxpy as cp

class IOP_Objective:
    '''
    The base class for IOP objectives
    '''
    def __init__ (self):
        # to save the objective value
        self._objective_expression = 0

    def getObjectiveValue(self):
        if isinstance(self._objective_expression, int):
            return self._objective_expression
        else:
            return self._objective_expression.value

    def addObjectiveValue(self, iop, objective_value):
        return objective_value

class IOP_Constraint(IOP_Objective):
    '''
    The base class for IOP constriant
    '''
    def addConstraints(self, iop, constraints):
        return constraints