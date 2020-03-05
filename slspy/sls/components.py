import numpy as np
import cvxpy as cp

class SLS_Objective:
    '''
    The base class for SLS objectives
    '''
    def __init__ (self):
        # to save the objective value
        self._objective_expression = 0

    def getObjectiveValue(self):
        if isinstance(self._objective_expression, int):
            return self._objective_expression
        else:
            return self._objective_expression.value

    def addObjectiveValue(self, sls, objective_value):
        return objective_value

class SLS_Constraint(SLS_Objective):
    '''
    The base class for SLS constriant
    '''
    def addConstraints(self, sls, constraints):
        return constraints