import numpy as np
import cvxpy as cp

def SLS_Objective_H2 (C1=None, D12=None, Phi_x=[],Phi_u=[]):
    if (C1 is None) or (D12 is None):
        return None
    #if len(Phi_x) != len(Phi_u): # should be handled already
    #    return None

    objective = 0
    for t in range(len(Phi_x)):
        vec_x = np.dot(C1, Phi_x[t])
        vec_u = np.dot(D12, Phi_u[t])
        objective = np.dot(vec_x,vec_x) + np.dot(vec_u,vec_u)
    
    return 0
