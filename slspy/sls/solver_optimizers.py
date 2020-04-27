from .components import SLS_SolverOptimizer
import cvxpy as cp
import numpy as np
from cvxpy.expressions.variable import Variable as CVX_Variable
from cvxpy.constraints.zero import Equality as CVX_Equality
from cvxpy.atoms.affine.index import index as CVX_index
from cvxpy.expressions.constants.constant import Constant as CVX_Constant
from cvxpy.atoms.affine.binary_operators import MulExpression as CVX_Multiplication
from cvxpy.atoms.affine.reshape import reshape as CVX_reshape

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

class SLS_SolOpt_VariableReduction (SLS_SolverOptimizer):
    assigned_variables = {}
    '''
    @staticmethod
    def getAssignedVariables(expression):
        if isinstance(expression,CVX_Multiplication):
            # handle nested multiplication
            return SLS_SolOpt_VariableReduction.expandMultiplication(expression)
        elif isinstance(expression,CVX_Variable):
            if expression not in SLS_SolOpt_VariableReduction.assigned_variables:
                # create variable
                rows = []
                for ix in range(expression.shape[0]):
                    row = []
                    for iy in range(expression.shape[1]):
                        row.append(cp.Variable(1))
                    rows.append(row)
                SLS_SolOpt_VariableReduction.assigned_variables[expression] = cp.bmat(rows)
            return SLS_SolOpt_VariableReduction.assigned_variables[expression]
        return expression

    @staticmethod
    def expandMultiplication(multiplication):
        
        #expand multiplication and eliminate zero terms

        #It turns out to be very inefficient. Avoid this trick.
        
        expression = SLS_SolOpt_VariableReduction.getAssignedVariables(multiplication.args[0])
        expression_constant = isinstance(expression,CVX_Constant)

        for argument_index in range(1,len(multiplication.args)):
            arg = SLS_SolOpt_VariableReduction.getAssignedVariables(multiplication.args[argument_index])
            arg_constant = isinstance(arg,CVX_Constant)
            # compute expression * arg
            rows = []
            for ix in range(expression.shape[0]):
                row = []
                for iy in range(arg.shape[1]):
                    dot_product = None
                    for iz in range(expression.shape[1]):
                        # only work for Constant
                        if expression_constant:
                            value_a = expression.value[ix, iz]
                            if value_a == 0.0:
                                continue
                        else:
                            value_a = expression.args[ix].args[iz]
                            if isinstance(value_a,CVX_reshape):
                                value_a = value_a.value
                                if value_a == 0.0:
                                    continue
                        # cannot check value_a == 0.0 here
                        # as value_a might be cp Variable

                        if arg_constant:
                            value_b = arg.value[iz, iy]
                            if value_b == 0.0:
                                continue
                        else:
                            value_b = arg.args[iz].args[iy]
                            if isinstance(value_b,CVX_reshape):
                                value_b = value_b.value
                                if value_b == 0.0:
                                    continue

                        product = value_a * value_b
                        if dot_product is None:
                            dot_product = product
                        else: 
                            dot_product += product
                    if dot_product is None:
                        dot_product = 0.0
                    #print(dot_product)
                    row.append(dot_product)
                rows.append(row)
            expression = cp.bmat(rows)
            expression_constant = isinstance(expression,CVX_Constant)
        return expression
    '''
    @staticmethod
    def expandArguments(argument_index, arguments):
        # this allows replacing an argument in the 'arguments' list
        expression = arguments[argument_index]
        if isinstance(expression,CVX_Constant):
            return
        if isinstance(expression,CVX_Variable):
            # replace the variable if it exists in assigned variables
            if expression in SLS_SolOpt_VariableReduction.assigned_variables:
                arguments[argument_index] = SLS_SolOpt_VariableReduction.assigned_variables[expression]
            return
        #if isinstance(expression,CVX_Multiplication):
        #    # flatten the multiplications
        #    arguments[argument_index] = SLS_SolOpt_VariableReduction.expandMultiplication(expression)
        #    return
        # expand
        for argument_index in range(len(expression.args)):
            SLS_SolOpt_VariableReduction.expandArguments(argument_index,expression.args)

    @staticmethod
    def expandExpression(expression):
        # expand
        for argument_index in range(len(expression.args)):
            SLS_SolOpt_VariableReduction.expandArguments(argument_index,expression.args)
    
    @staticmethod
    def optimize(objective_value, constraints):
        # parse and expand all constraints
        reduced_objective_value = None
        reduced_constraints = []

        # first check if the constraint is an assignment
        # remember the assignment of variables
        SLS_SolOpt_VariableReduction.assigned_variables = {}

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
                    value = constraint.args[0].value
                
                if isinstance(constraint.args[1], CVX_Variable):
                    variable = constraint.args[1]
                if isinstance(constraint.args[1], CVX_index):
                    index = constraint.args[1]
                if isinstance(constraint.args[1], CVX_Constant):
                    value = constraint.args[1].value

                if value is not None:
                    if variable is not None:
                        # it is an assignment
                        if variable in SLS_SolOpt_VariableReduction.assigned_variables.keys():
                            # have to check if there exist two conflict assignments
                            if value != SLS_SolOpt_VariableReduction.assigned_variables[variable]:
                                # conflict assignment
                                return 'infeasible', objective_value, constraints
                        else:
                            SLS_SolOpt_VariableReduction.assigned_variables[variable] = value
                        continue

                    if index is not None:
                        # get the corresponding variable
                        variable = index.args[0]
                        if variable in SLS_SolOpt_VariableReduction.assigned_variables.keys():
                            # have to check if there exist two conflict assignments
                            assigned_value = SLS_SolOpt_VariableReduction.assigned_variables[variable][index.key[0],index.key[1]]
                            if assigned_value[0,0] is not None:
                                if value != assigned_value:
                                    # conflict assignment
                                    return 'infeasible', objective_value, constraints
                            else:
                                SLS_SolOpt_VariableReduction.assigned_variables[variable][index.key[0],index.key[1]] = value
                        else:
                            SLS_SolOpt_VariableReduction.assigned_variables[variable] = np.full(variable.shape, None)
                            SLS_SolOpt_VariableReduction.assigned_variables[variable][index.key[0],index.key[1]] = value
                        continue

            # not a simple assignment:
            reduced_constraints.append(constraint)

        # organize assigned variables: replace None by cvxpy variable, and make it CVX_Constant if all the values are defined?
        for variable in SLS_SolOpt_VariableReduction.assigned_variables.keys():
            value = SLS_SolOpt_VariableReduction.assigned_variables[variable]

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
                SLS_SolOpt_VariableReduction.assigned_variables[variable] = cp.bmat(rows)
            else:
                SLS_SolOpt_VariableReduction.assigned_variables[variable] = CVX_Constant(value=value)

        # expand the all multiplications in args
        SLS_SolOpt_VariableReduction.expandExpression(reduced_objective_value)

        for constraint in reduced_constraints:
            SLS_SolOpt_VariableReduction.expandExpression(constraint)

        return 'success', reduced_objective_value, reduced_constraints

    @staticmethod
    def postProcess():
        for variable in SLS_SolOpt_VariableReduction.assigned_variables.keys():
            variable.value = SLS_SolOpt_VariableReduction.assigned_variables[variable].value