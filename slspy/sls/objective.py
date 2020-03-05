from .components import SLS_Objective
import numpy as np
import cvxpy as cp
'''
To create a new SLS objective, inherit the following base function and customize the specified methods.

class SLS_Objective:
    def __init__ (self):
    def getObjectiveValue(self):
        return self._objective_expression
    def addObjectiveValue(self, sls, objective_value):
        return objective_value
'''

class SLS_Obj_H2(SLS_Objective):
    '''
    return 
        || [C1, D12][Phi_x; Phi_u] ||_H2^2
        for state-feedback SLS
        || [C1, D12][Phi_xx, Phi_xy; Phi_ux, Phi_uy][B1; D21]||_H2^2
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
            Phi_xx = sls._Phi_xx
            Phi_ux = sls._Phi_ux
            Phi_xy = sls._Phi_xy
            Phi_uy = sls._Phi_uy
            for tau in range(len(Phi_xx)):
                self._objective_expression += cp.sum_squares(
                    C1 *Phi_xx[tau]*B1 +
                    D12*Phi_ux[tau]*B1 +
                    C1 *Phi_xy[tau]*D21 + 
                    D12*Phi_uy[tau]*D21
                )

        return objective_value + self._objective_expression

class SLS_Obj_LQ(SLS_Objective):
    '''
    This function assumes 
    1) all disturbance / noise is zero-centered Gaussian
    2) process disturbance and measurement noise are uncorrelated
                    
    Cov_w is the covariance matrix for process disturbance
    Cov_v is the covariance matrix for measurement noise
    
    return
        || [Q^0.5, R^0.5][Phi_x; Phi_u]Cov_w^0.5 ||_Frob^2
        i.e. LQR for state-feedback SLS

        || [Q^0.5, R^0.5][Phi_xx, Phi_xy; Phi_ux, Phi_uy][Cov_w^0.5; Cov_v^0.5]||_Frob^2
        i.e. LQG for output-feedback SLS
    '''
    
    def __init__(self, QSqrt=None, RSqrt=None, Cov_wSqrt=None, Cov_vSqrt=None):   
        self._QSqrt = QSqrt        
        self._RSqrt = RSqrt
        self._Cov_wSqrt = Cov_wSqrt
        self._Cov_vSqrt = Cov_vSqrt

    def addObjectiveValue(self, sls, objective_value):
        QSqrt = self._QSqrt 
        RSqrt = self._RSqrt
        Cov_wSqrt = self._Cov_wSqrt
        Cov_vSqrt = self._Cov_vSqrt
        
        # default values
        if QSqrt is None:
            QSqrt = sls._system_model._C1
        if RSqrt is None:
            RSqrt = sls._system_model._D12 
        if Cov_wSqrt is None:
            Cov_wSqrt = sls._system_model._B1 
        if Cov_vSqrt is None:
            Cov_vSqrt = sls._system_model._D21

        self._objective_expression = 0
        if sls._state_feedback:
            # state-feedback
            Phi_x = sls._Phi_x
            Phi_u = sls._Phi_u
            for tau in range(len(Phi_x)):
                self._objective_expression += cp.sum_squares(QSqrt * Phi_x[tau] * Cov_wSqrt + 
                                                             RSqrt * Phi_u[tau] * Cov_wSqrt)
        else:
            # output-feedback
            Phi_xx = sls._Phi_xx
            Phi_ux = sls._Phi_ux
            Phi_xy = sls._Phi_xy
            Phi_uy = sls._Phi_uy

            for tau in range(len(Phi_xx)):
                self._objective_expression += cp.sum_squares(
                    QSqrt * Phi_xx[tau] * Cov_wSqrt +
                    RSqrt * Phi_ux[tau] * Cov_wSqrt +
                    QSqrt * Phi_xy[tau] * Cov_vSqrt + 
                    RSqrt * Phi_uy[tau] * Cov_vSqrt
                )

        return objective_value + self._objective_expression
    
class SLS_Obj_HInf(SLS_Objective):
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

class SLS_Obj_L1(SLS_Objective):
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

class SLS_Obj_RFD(SLS_Objective):
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
