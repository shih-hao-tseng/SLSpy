from .components import SLS_Solver, SLS_SolverOptimizer
import cvxpy as cp
'''
To create a new SLS solver, inherit the following base function and customize the specified methods.

class SLS_Solver:
    def __init__ (self, sls, optimizers=[]):
        pass
    def solve (
        self,
        controller,
        objective_value,
        constraints
    ):
        return controller

To create a new solver optimizer, inherit the following base function and customize the specified methods.

class SLS_SolverOptimizer:
    @staticmethod
    def optimize(objective_value, constraints):
        return objective_value, constraints
'''

class SLS_SolOpt_ReduceRedundancy (SLS_SolverOptimizer):
    @staticmethod
    def optimize(objective_value, constraints):
        return objective_value, constraints

class SLS_Sol_CVX:
    def __init__ (self, sls, optimizers=[SLS_SolOpt_ReduceRedundancy]):
        self._sls = sls
        self._sls_problem = None #cp.Problem(cp.Minimize(0))
        self._solver_optimizers = optimizers

    def get_SLS_Problem (self):
        return self._sls_problem

    def solve (
        self,
        objective_value,
        constraints
    ):
        for sol_opt in self._solver_optimizers:
            # apply the optimizers
            objective_value, constraints = sol_opt.optimize(objective_value, constraints)

        self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        self._sls_problem.solve()

        problem_value = self._sls_problem.value
        solver_status = self._sls_problem.status 

        return problem_value, solver_status