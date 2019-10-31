import numpy as np
import cvxpy as cp
from scipy.linalg import block_diag as blkdiag

def SLS_Objective_Value_H2 (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return || [C1, D12][Phi_x; Phi_u] ||_H2^2
    '''
    objective_value = 0
    for tau in range(len(Phi_x)):
        objective_value += cp.sum_squares(C1  * Phi_x[tau] + D12 * Phi_u[tau])
    
    return objective_value

def SLS_Objective_Value_HInf (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return max singular value of [C1,D12][R;M]
    '''
    matrix = []

    horizon = len(Phi_x)
    block_rows = C1.shape[0]
    if horizon > 0:
        block_cols = Phi_x[0].shape[1]
        block = np.zeros([block_rows,block_cols])

    for tau in range(horizon):
        row = []
        for times in range (tau):
            row.append(block)
        row.append(C1*Phi_x[tau] + D12*Phi_u[tau])
        for times in range (horizon - tau - 1):
            row.append(block)

        matrix.append(row)

    block_diagonal_matrix = cp.bmat(matrix)

    return cp.sigma_max(block_diagonal_matrix)

def SLS_Objective_Value_L1 (C1, D12, Phi_x=[], Phi_u=[]):
    '''
    return max row sum of [C1,D12][R;M]
    '''
    matrix = []

    horizon = len(Phi_x)
    block_rows = C1.shape[0]
    if horizon > 0:
        block_cols = Phi_x[0].shape[1]
        block = np.zeros([block_rows,block_cols])

    for tau in range(horizon):
        row = []
        for times in range (tau):
            row.append(block)
        row.append(C1*Phi_x[tau] + D12*Phi_u[tau])
        for times in range (horizon - tau - 1):
            row.append(block)

        matrix.append(row)
    
    block_diagonal_matrix = cp.bmat(matrix)

    return cp.norm(block_diagonal_matrix,'inf')