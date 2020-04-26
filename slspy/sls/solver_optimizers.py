from .components import SLS_SolverOptimizer
import cvxpy as cp
import numpy as np
from cvxpy.expressions.variable import Variable as CVX_Variable
from cvxpy.constraints.zero import Equality as CVX_Equality
from cvxpy.atoms.affine.index import index as CVX_index
from cvxpy.expressions.constants.constant import Constant as CVX_Constant
from cvxpy.atoms.affine.binary_operators import MulExpression as CVX_Multiplication

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
    def expandExpression(expression):
        print(type(expression))
        print(expression)
        for arg in expression.args:
            if isinstance(arg,CVX_Constant):
                print('const')
                continue
            if isinstance(arg,CVX_Variable):
                print('varible')
                continue
            SLS_SolOpt_ReduceRedundancy.expandExpression(arg)
    
    @staticmethod
    def optimize(objective_value, constraints):
        # parse and expand all constraints
        reduced_objective_value = None
        reduced_constraints = []

        # first check if the constraint is an assignment
        # remember the assignment of variables
        assigned_variables = {}

        for item in objective_value.args:
            if isinstance(item, CVX_Constant):
                # constant does not matter
                continue
            if reduced_objective_value is None:
                reduced_objective_value = item
            else:
                reduced_objective_value += item

        for constraint in constraints:
            # reduce the assignment constraints
            if isinstance(constraint, CVX_Equality):
                # handle equality: args[0] == args[1]
                variable = None
                index = None
                value = None

                if isinstance(constraint.args[0], CVX_Variable):
                    variable = constraint.args[0]
                if isinstance(constraint.args[0], CVX_index):
                    index = constraint.args[0]
                if isinstance(constraint.args[0], CVX_Constant):
                    value = constraint.args[0]
                
                if isinstance(constraint.args[1], CVX_Variable):
                    variable = constraint.args[1]
                if isinstance(constraint.args[1], CVX_index):
                    index = constraint.args[1]
                if isinstance(constraint.args[1], CVX_Constant):
                    value = constraint.args[1].value

                if (variable is not None) and (value is not None):
                    # it is an assignment
                    # print ('assign %s == %s' %(variable,value))
                    if variable in assigned_variables.keys():
                        # have to check if there exist two conflict assignments
                        if value != assigned_variables[variable]:
                            # conflict assignment
                            return 'infeasible', objective_value, constraints
                    else:
                        assigned_variables[variable] = value
                        variable.value = value
                    continue

                if (index is not None) and (value is not None):
                    # get the corresponding variable
                    # print ('assign %s == %s' %(index,value))
                    variable = index.args[0]
                    if variable in assigned_variables.keys():
                        # have to check if there exist two conflict assignments
                        assigned_value = assigned_variables[variable][index.key[0],index.key[1]]
                        if assigned_value[0,0] is not None:
                            if value != assigned_value:
                                # conflict assignment
                                return 'infeasible', objective_value, constraints
                    else:
                        assigned_variables[variable] = np.full(variable.shape, None)
                        assigned_variables[variable][index.key[0],index.key[1]] = value
                        if variable.value is None:
                            variable.value = np.empty(variable.shape)
                        variable.value[index.key[0],index.key[1]] = value
                    continue

            # not a simple assignment:
            reduced_constraints.append(constraint)

        for constraint in reduced_constraints:
            print(constraint)

        # expand the all multiplications in args
        SLS_SolOpt_ReduceRedundancy.expandExpression(reduced_objective_value)

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