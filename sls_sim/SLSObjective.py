import numpy as np
import cvxpy as cp

class SLSObjective:
    '''
    The base class for SLS objectives
    '''
    def addObjectiveValue(self, sls, objective_value):
        return objective_value

class SLSObj_H2(SLSObjective):
    '''
    return || [C1, D12][Phi_x; Phi_u] ||_H2^2
    '''
    def addObjectiveValue(self, sls, objective_value):
        C1  = sls._system_model._C1
        D12 = sls._system_model._D12
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u
        
        for tau in range(len(Phi_x)):
            objective_value += cp.sum_squares(C1*Phi_x[tau] + D12*Phi_u[tau])

        return objective_value

class SLSObj_HInf(SLSObjective):
    '''
    return max singular value of [C1,D12][R;M]
    '''
    def addObjectiveValue(self, sls, objective_value):
        C1  = sls._system_model._C1
        D12 = sls._system_model._D12
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u

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

        return objective_value + cp.sigma_max(block_diagonal_matrix)

class SLSObj_L1(SLSObjective):
    '''
    return max row sum of [C1,D12][R;M]
    '''
    def addObjectiveValue(self, sls, objective_value):
        C1  = sls._system_model._C1
        D12 = sls._system_model._D12
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u

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

        return objective_value + cp.norm(block_diagonal_matrix,'inf')

class SLSObj_RFD(SLSObjective):
    '''
    regularization for design (RFD) objective
    '''
    def __init__ (self,rfdCoeff=0):
        self._rfdCoeff = rfdCoeff
        self._acts_rfd = []

    def addObjectiveValue(self, sls, objective_value):
        actPenalty = 0
        self._acts_rfd = []

        for i in range (sls._system_model._Nu):
            Phi_u_i = []
            for t in range (sls._FIR_horizon):
                u = sls._Phi_u[t][i,:]
                Phi_u_i.append(u)
                if t == 0:
                    self._acts_rfd.append(cp.norm(u,2))

            actPenalty += cp.norm(cp.bmat(Phi_u_i),2)

        return objective_value + self._rfdCoeff * actPenalty 

    def getActsRFD (self):
        tol = 1e-4
        
        acts = []
        for i in range(self._acts_rfd):
            if self._acts_rfd[i].value > tol:
                acts.append(i)

        return acts
