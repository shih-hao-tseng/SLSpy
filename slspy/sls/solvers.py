from .components import SLS_Solver, SLS_SolverOptimizer
from .solver_optimizers import SLS_SolOpt_ReduceRedundancy
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
'''

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
            if issubclass(sol_opt, SLS_SolverOptimizer):
                solver_status, objective_value, constraints = sol_opt.optimize(objective_value, constraints)
                if solver_status == 'infeasible':
                    return 0.0, solver_status

        self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        self._sls_problem.solve()

        problem_value = self._sls_problem.value
        solver_status = self._sls_problem.status 

        for sol_opt in self._solver_optimizers:
            # optimizers post-process
            if issubclass(sol_opt, SLS_SolverOptimizer):
                sol_opt.postProcess()

        return problem_value, solver_status