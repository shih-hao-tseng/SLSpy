from .components import SLS_SolverOptimizer
import cvxpy as cp
import numpy as np
from cvxpy.expressions.variable import Variable as CVX_Variable
from cvxpy.constraints.zero import Equality as CVX_Equality
from cvxpy.atoms.affine.index import index as CVX_index
from cvxpy.expressions.constants.constant import Constant as CVX_Constant

'''
To create a new solver optimizer, inherit the following base function and customize the specified methods.

class SLS_SolverOptimizer:
    @staticmethod
    def optimize(objective_value, constraints):
        return status, objective_value, constraints
'''

class SLS_SolOpt_ReduceRedundancy (SLS_SolverOptimizer):
    @staticmethod
    def expandMultiplication(assigned_variables, multi):
        pass
    
    @staticmethod
    def optimize(objective_value, constraints):
        #TODO
        '''
        x = cp.Variable(1)
        y = np.zeros([1,1])
        #z = np.concatenate(x,y)
        cons = [ cp.bmat([x,y]) == np.zeros([2,1]) ]
        print(cons)
        for con in cons:
            print(con)
            con.args[0] = cp.bmat([x,y + x])
            print(con)
        '''

        # parse and expand all constraints
        reduced_objective_value = 0.0
        reduced_constraints = []

        '''
        for arg in objective_value.args:
            print(arg)
            print(type(arg))
            if not isinstance(arg,CVX_Constant):
                # remove constants
                reduced_objective_value += arg
    
        objective_value = reduced_objective_value
        '''

        # first check if the constraint is an assignment
        # remember the assignment of variables
        assigned_variables = {}

        for constraint in constraints:
            if isinstance(constraint, CVX_Equality):
                # equality: args[0] == args[1]
                print('is equality')
                print(type(constraint.args[0]))
                print(type(constraint.args[1]))
                variable = None
                index = None
                value = None

                if isinstance(constraint.args[0], CVX_Equality):
                    variable = constraint.args[0]
                if isinstance(constraint.args[0], CVX_index):
                    index = constraint.args[0]
                if isinstance(constraint.args[0], CVX_Constant):
                    value = constraint.args[0]
                
                if isinstance(constraint.args[1], CVX_Equality):
                    variable = constraint.args[1]
                if isinstance(constraint.args[1], CVX_index):
                    index = constraint.args[1]
                if isinstance(constraint.args[1], CVX_Constant):
                    value = constraint.args[1]

                if (variable is not None) and (value is not None):
                    # it is an assignment
                    print('assign %s = %s' % (variable,value))
                    if variable in assigned_variables.keys():
                        # have to check if there exist two conflict assignments
                        if value != assigned_variables[variable]:
                            # conflict assignment
                            return 'infeasible', objective_value, constraints
                    else:
                        assigned_variables[variable] = value    
                        variable.value = value.value
                    continue

                if (index is not None) and (value is not None):
                    print('assign %s = %s' % (index,value))
                    variable = index
                    assigned_variables[variable] = value
                    print(type(value.value))
                    print(type(variable.value))
                    continue


            




        '''
            print (type(constraint))
            print (len(constraint.args))


            for arg in constraint.args:
                print(arg)
                print(arg.shape)
                pass


            if isinstance(constraint.args[0], CVX_Variable):
                print ('is variable')
                pass
            print(constraint.args[0])
            print(constraint.args[0].id)
            print(type(constraint.args[0]))
            print(constraint.args[1])
            print(type(constraint.args[1]))

            print('---')
            pass
        '''
        return 'success', objective_value, constraints