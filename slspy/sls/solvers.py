from .components import SLS_Solver, SLS_SolverOptimizer
from .solver_optimizers import SLS_SolOpt_VariableReduction
import cvxpy as cp
#import time

'''
To create a new SLS solver, inherit the following base function and customize the specified methods.

class SLS_Solver:
    def __init__ (self, sls, optimizers=[]):
        pass
    def solve (
        self,
        objective_value,
        constraints
    ):
        return problem_value, solver_status
'''

class SLS_Sol_CVX:
    def __init__ (self, sls, optimizers=[SLS_SolOpt_VariableReduction]):
        self._sls = sls
        self._sls_problem = None #cp.Problem(cp.Minimize(0))
        self._solver_optimizers = []
        for sol_opt in optimizers:
            if issubclass(sol_opt, SLS_SolverOptimizer):
                self._solver_optimizers.append(sol_opt)
        self._CVX_solver = None

    def get_SLS_Problem (self):
        return self._sls_problem

    def set_CVX_Solver(self, CVX_solver):
        self._CVX_solver = CVX_solver

    def solve (
        self,
        objective_value,
        constraints
    ):
        #time_start = time.perf_counter()
        #self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        #self._sls_problem.solve()
        #time_end   = time.perf_counter()
        #print(" without optimization %.8f, " % (time_end-time_start))

        #time_start = time.perf_counter()
        for sol_opt in self._solver_optimizers:
            # apply the optimizers
            solver_status, objective_value, constraints = sol_opt.optimize(objective_value, constraints)
            if solver_status == 'infeasible':
                return 0.0, solver_status

        self._sls_problem = cp.Problem (cp.Minimize(objective_value), constraints)
        if self._CVX_solver is not None:
            self._sls_problem.solve(solver=self._CVX_solver)
        else:
            self._sls_problem.solve()

        for sol_opt in self._solver_optimizers:
            # optimizers post-process
            sol_opt.postProcess()
        #time_end   = time.perf_counter()
        #print(" with optimization %.8f, " % (time_end-time_start))

        problem_value = self._sls_problem.value
        solver_status = self._sls_problem.status 

        return problem_value, solver_status