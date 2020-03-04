from .components import SLSObjective
import numpy as np
import cvxpy as cp
'''
To create a new SLS objective, inherit the following base function and customize the specified methods.

class SLSObjective:
    def __init__ (self):
    def getObjectiveValue(self):
        return self._objective_expression
    def addObjectiveValue(self, sls, objective_value):
        return objective_value
'''

class SLSObj_H2(SLSObjective):
    '''
    return 
        || [C1, D12][Phi_x; Phi_u] ||_H2^2
        for state-feedback SLS
        || [C1, D12][Phi_xx, Phi_xy; Phi_ux, Phi_uy][B1; D21] + D11 ||_H2^2
        for output-feedback SLS
    '''
    def addObjectiveValue(self, sls, objective_value):
        C1  = sls._system_model._C1
        D12 = sls._system_model._D12

        self._objective_expression = 0
        if sls._state_feedback:
            # state-feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u
            for tau in range(len(Phi_x)):
                self._objective_expression += cp.sum_squares(C1*Phi_x[tau] + D12*Phi_u[tau])
        else:
            # output-feedback
            B1  = sls._system_model._B1
            D21 = sls._system_model._D21
            D11 = sls._system_model._D11
            Phi_xx = sls._Phi_xx
            Phi_ux = sls._Phi_ux
            Phi_xy = sls._Phi_xy
            Phi_uy = sls._Phi_uy
            for tau in range(len(Phi_xx)):
                self._objective_expression += cp.sum_squares(
                    C1 *Phi_xx[tau]*B1 +
                    D12*Phi_ux[tau]*B1 +
                    C1 *Phi_xy[tau]*D21 + 
                    D12*Phi_uy[tau]*D21 +
                    D11
                )

        return objective_value + self._objective_expression

class SLSObj_HInf(SLSObjective):
    '''
    return max singular value of [C1,D12][R;M]
    '''
    # not yet support output feedback
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
    # not yet support output feedback
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
    # not yet support output feedback
    def __init__ (self,rfdCoeff=0):
        self._rfdCoeff = rfdCoeff
        self._acts_rfd = []

    def addObjectiveValue(self, sls, objective_value):
        actPenalty = 0
        self._acts_rfd = []
        for i in range (sls._system_model._Nu):
            Phi_u_i = []
            self._acts_rfd.append(cp.norm(sls._Phi_u[1][i,:],2))
            for t in range (1,sls._FIR_horizon+1):
                Phi_u_i.append(sls._Phi_u[t][i,:])

            actPenalty += cp.norm(cp.bmat([Phi_u_i]),2)

        self._objective_expression = self._rfdCoeff * actPenalty

        return objective_value + self._objective_expression

    def getActsRFD (self):
        tol = 1e-4

        acts = []
        for i in range(len(self._acts_rfd)):
            if self._acts_rfd[i].value > tol:
                acts.append(i)

        return acts
