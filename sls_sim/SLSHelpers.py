import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag as blkdiag

def SLS_Objective_Value_H2 (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return || [C1, D12][Phi_x; Phi_u] ||_H2^2
    '''
    objective_value = 0
    for tau in range(len(Phi_x)):
        objective_value += cp.sum_squares(C1  * Phi_x[tau])
        objective_value += cp.sum_squares(D12 * Phi_u[tau])
    
    return objective_value

def SLS_Objective_Value_HInf (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return max singular value of [C1,D12][R;M]
    '''
    #matrix = []
    #
    #for tau in range(len(Phi_x)):
    #    matrix = blkdiag(matrix, C1  * Phi_x[tau] + D12 * Phi_u[tau])
    #
    #return cp.sigma_max(matrix)
    return None

def SLS_Objective_Value_L1 (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return max row sum of [C1,D12][R;M]
    '''
    return None