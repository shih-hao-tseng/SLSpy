import numpy as np
import cvxpy as cp

def SLS_Objective_Value_H2 (C1=None, D12=None, Phi_x=[], Phi_u=[]):
    '''
    return || [C1, D12][Phi_x; Phi_u] ||_H2^2
    '''
    if (C1 is None) or (D12 is None):
        return None
    #if len(Phi_x) != len(Phi_u): # should be handled already
    #    return None

    objective_value = 0
    for tau in range(len(Phi_x)):
        objective_value += cp.sum_squares(C1  * Phi_x[tau])
        objective_value += cp.sum_squares(D12 * Phi_u[tau])
    
    return objective_value
