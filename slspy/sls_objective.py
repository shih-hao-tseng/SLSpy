import numpy as np
import cvxpy as cp

class SLSObjective:
    '''
    The base class for SLS objectives
    '''
    def __init__ (self):
        # to save the objective value
        self._objective_expression = 0

    def getObjectiveValue(self):
        return self._objective_expression.value

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

        self._objective_expression = 0
        for tau in range(len(Phi_x)):
            self._objective_expression += cp.sum_squares(C1*Phi_x[tau] + D12*Phi_u[tau])

        return objective_value + self._objective_expression

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
        self._objective_expression = cp.sigma_max(block_diagonal_matrix)

        return objective_value + self._objective_expression

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
        self._objective_expression = cp.norm(block_diagonal_matrix,'inf')

        return objective_value + self._objective_expression

class SLSObj_RFD(SLSObjective):
    '''
    regularization for design (RFD) objective
    '''
    def __init__ (self,rfdCoeff=0):
        self._rfdCoeff = rfdCoeff
        self._acts_rfd = None
        self._reference_system = None

    def addObjectiveValue(self, sls, objective_value):
        actPenalty = 0
        
        # for higher performance, don't keep generating variables
        if self._reference_system is not sls._system_model:
            # for a new system
            self._reference_system = sls._system_model
            self._acts_rfd = []
            for i in range (sls._system_model._Nu):
                u = sls._Phi_u[0][i,:]
                self._acts_rfd.append(cp.norm(u,2))

        for i in range (sls._system_model._Nu):
            Phi_u_i = []

            for t in range (sls._FIR_horizon):
                Phi_u_i.append(sls._Phi_u[t][i,:])

            actPenalty += cp.norm(cp.bmat(Phi_u_i),2)

        self._objective_expression = self._rfdCoeff * actPenalty

        return objective_value + self._objective_expression

    def getActsRFD (self):
        tol = 1e-4
        
        if self._acts_rfd is None:
            return []

        acts = []
        for i in range(len(self._acts_rfd)):
            if self._acts_rfd[i].value > tol:
                acts.append(i)

        return acts
