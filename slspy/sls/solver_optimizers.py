from .components import SLS_SolverOptimizer
import cvxpy as cp
from cvxpy.expressions.variable import Variable as CVX_Variable

'''
To create a new solver optimizer, inherit the following base function and customize the specified methods.

class SLS_SolverOptimizer:
    @staticmethod
    def optimize(objective_value, constraints):
        return objective_value, constraints
'''

class SLS_SolOpt_ReduceRedundancy (SLS_SolverOptimizer):
    @staticmethod
    def optimize(objective_value, constraints):
        #TODO
        for constraint in constraints:
            print(constraint)
            print(type(constraint))
            print('===')
            if isinstance(constraint.args[0], CVX_Variable):
                # Handle variable
                print(constraint.args[0].shape)
                print ('is variable')
            print(constraint.args[0])
            print(constraint.args[0].id)
            print(type(constraint.args[0]))
            print(constraint.args[1])
            print(type(constraint.args[1]))

            print('---')
            pass
        return objective_value, constraints