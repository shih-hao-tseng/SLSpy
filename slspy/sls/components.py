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

class SLS_Solver:
    '''
    The base class for SLS solver
    A solver takes the objective and constraints to solve the SLS problem and generate the controller 
    '''
    def __init__ (self, sls, optimizers=[]):
        # solvers might need to alter _Phi_x, _Phi_u, directly
        self._sls = sls
        self._solver_optimizers = optimizers
        pass
    def solve (
        self,
        objective_value,
        constraints
    ):
        '''
        status: string return by the solver
        '''
        problem_value = 0.0
        solver_status = 'feasible'
        return problem_value, solver_status
    
class SLS_SolverOptimizer:
    '''
    The base class for a solver optimizer
    The optimizer tries to simplify the problem before feeding it to the solver so that it is more efficient to solve
    '''
    @staticmethod
    def optimize(objective_value, constraints):
        # status: the issues detected by the optimizer, e.g., 'success', 'infeasible', etc.
        return status, objective_value, constraints