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

    @staticmethod
    def postProcess():
        pass
'''

class SLS_SolOpt_ReduceRedundancy (SLS_SolverOptimizer):
    assigned_variables = {}

    @staticmethod
    def expandMultiplication(multiplication):
        
        pass
    
    @staticmethod
    def expandArguments(argument_index, arguments):
        # this allows replacing an argument in the 'arguments' list
        expression = arguments[argument_index]
        if isinstance(expression,CVX_Constant):
            return
        if isinstance(expression,CVX_Variable):
            # replace the variable if it exists in assigned variables
            if expression in SLS_SolOpt_ReduceRedundancy.assigned_variables:
                arguments[argument_index] = SLS_SolOpt_ReduceRedundancy.assigned_variables[expression]
            return
        #if isinstance(expression,CVX_Multiplication):
        #    print('multiplication')
        #    print(expression)
        #    SLS_SolOpt_ReduceRedundancy.expandMultiplication(expression)
        #    return
        # expand
        for argument_index in range(len(expression.args)):
            SLS_SolOpt_ReduceRedundancy.expandArguments(argument_index,expression.args)

    @staticmethod
    def expandExpression(expression):
        # expand
        for argument_index in range(len(expression.args)):
            SLS_SolOpt_ReduceRedundancy.expandArguments(argument_index,expression.args)
    
    @staticmethod
    def optimize(objective_value, constraints):
        # parse and expand all constraints
        reduced_objective_value = None
        reduced_constraints = []

        # first check if the constraint is an assignment
        # remember the assignment of variables
        SLS_SolOpt_ReduceRedundancy.assigned_variables = {}

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
                    if variable in SLS_SolOpt_ReduceRedundancy.assigned_variables.keys():
                        # have to check if there exist two conflict assignments
                        if value != SLS_SolOpt_ReduceRedundancy.assigned_variables[variable]:
                            # conflict assignment
                            return 'infeasible', objective_value, constraints
                    else:
                        SLS_SolOpt_ReduceRedundancy.assigned_variables[variable] = value
                        variable.value = value
                    continue

                if (index is not None) and (value is not None):
                    # get the corresponding variable
                    # print ('assign %s == %s' %(index,value))
                    variable = index.args[0]
                    if variable in SLS_SolOpt_ReduceRedundancy.assigned_variables.keys():
                        # have to check if there exist two conflict assignments
                        assigned_value = SLS_SolOpt_ReduceRedundancy.assigned_variables[variable][index.key[0],index.key[1]]
                        if assigned_value[0,0] is not None:
                            if value != assigned_value:
                                # conflict assignment
                                return 'infeasible', objective_value, constraints
                    else:
                        SLS_SolOpt_ReduceRedundancy.assigned_variables[variable] = np.full(variable.shape, None)
                        SLS_SolOpt_ReduceRedundancy.assigned_variables[variable][index.key[0],index.key[1]] = value
                        if variable.value is None:
                            variable.value = np.empty(variable.shape)
                        variable.value[index.key[0],index.key[1]] = value
                    continue

            # not a simple assignment:
            reduced_constraints.append(constraint)

        # organize assigned variables: replace None by cvxpy variable, and make it CVX_Constant if all the values are defined?
        for variable in SLS_SolOpt_ReduceRedundancy.assigned_variables.keys():
            value = SLS_SolOpt_ReduceRedundancy.assigned_variables[variable]

            if None in value:
                rows = []
                for ix in range(value.shape[0]):
                    row = []
                    for iy in range(value.shape[1]):
                        if value[ix,iy] is None:
                            row.append(cp.Variable(1))
                        else:
                            row.append(value[ix,iy])
                    rows.append(row)
                SLS_SolOpt_ReduceRedundancy.assigned_variables[variable] = cp.bmat(rows)
            else:
                SLS_SolOpt_ReduceRedundancy.assigned_variables[variable] = CVX_Constant(value=value)

        # expand the all multiplications in args
        SLS_SolOpt_ReduceRedundancy.expandExpression(reduced_objective_value)

        for constraint in reduced_constraints:
            SLS_SolOpt_ReduceRedundancy.expandExpression(constraint)

        return 'success', reduced_objective_value, reduced_constraints

    @staticmethod
    def postProcess():
        for variable in SLS_SolOpt_ReduceRedundancy.assigned_variables.keys():
            variable.value = SLS_SolOpt_ReduceRedundancy.assigned_variables[variable].value