from .components import SLS_Solver
import cvxpy as cp
'''
To create a new SLS solver, inherit the following base function and customize the specified methods.

class SLS_Solver:
    def __init__ (self, sls):
        pass
    def solve (
        self,
        controller,
        objective_value,
        constraints
    ):
        return controller
'''

class SLS_Sol_CVX:
    def __init__ (self, sls):
        self._sls = sls
        self._sls_problem = None #cp.Problem(cp.Minimize(0))

    def get_SLS_Problem (self):
        return self._sls_problem

    def solve (
        self,
        objective_value,
        constraints
    ):
        self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        self._sls_problem.solve()

        problem_value = self._sls_problem.value
        solver_status = self._sls_problem.status 

        return problem_value, solver_status