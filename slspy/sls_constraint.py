
from sls_objective import SLSObjective
import cvxpy as cp
import numpy as np

class SLSConstraint(SLSObjective):
    '''
    The base class for SLS constriant
    '''
    def addConstraints(self, sls, constraints):
        return constraints

class SLSCons_SLS (SLSConstraint):
    '''
    The descrete-time SLS constrains
    '''
    def __init__(self, state_feedback=False):
        self._state_feedback = state_feedback

    def addConstraints(self, sls, constraints=[]):
        '''
        state-feedback constraints:
        [ zI-A, -B2 ][ Phi_x ] = I
                     [ Phi_u ]

        output-feedback constriants:
        [ zI-A, -B2 ][ Phi_xx Phi_xy ] = [ I 0 ]
                     [ Phi_ux Phi_uy ]
        [ Phi_xx Phi_xy ][ zI-A ] = [ I ]
        [ Phi_ux Phi_uy ][ -C2  ]   [ 0 ]
        '''
        Nx = sls._system_model._Nx
        Nu = sls._system_model._Nu

        # sls constraints
        # the below constraints work for output-feedback case as well because
        # sls._Phi_x = sls._Phi_xx and sls._Phi_u = sls._Phi_ux
        constraints += [ sls._Phi_x[0] == np.eye(Nx) ]
        constraints += [ 
            (sls._system_model._A  * sls._Phi_x[sls._FIR_horizon-1] +
             sls._system_model._B2 * sls._Phi_u[sls._FIR_horizon-1] ) == np.zeros([Nx, Nx]) 
        ]
        for tau in range(sls._FIR_horizon-1):
            constraints += [
                sls._Phi_x[tau+1] == (
                    sls._system_model._A  * sls._Phi_x[tau] +
                    sls._system_model._B2 * sls._Phi_u[tau]
                )
            ]

        if not self._state_feedback:
            # output-feedback constraints
            constraints += [
                sls._Phi_xy[0] == sls._system_model._B2 * sls._Phi_uy[0]
            ]
            constraints += [ 
                (sls._system_model._A  * sls._Phi_xy[sls._FIR_horizon-1] +
                 sls._system_model._B2 * sls._Phi_uy[sls._FIR_horizon  ]) == np.zeros([Nx, Ny])
            ]
            constraints += [ 
                (sls._Phi_xx[sls._FIR_horizon-1] * sls._system_model._A  +
                 sls._Phi_xy[sls._FIR_horizon-1] * sls._system_model._C2 ) == np.zeros([Nx, Nx])
            ]
            constraints += [
                sls._Phi_ux[0] == sls._Phi_uy[0] * sls._system_model._C2
            ]
            constraints += [
                (sls._Phi_ux[sls._FIR_horizon-1] * sls._system_model._A  +
                 sls._Phi_uy[sls._FIR_horizon  ] * sls._system_model._C2 ) == np.zeros([Nu, Nx])
            ]
            for tau in range(sls._FIR_horizon-1):
                constraints += [ 
                    sls._Phi_xy[tau+1] == (
                        sls._system_model._A  * sls._Phi_xy[tau] +
                        sls._system_model._B2 * sls._Phi_uy[tau+1]
                    )
                ]

                constraints += [
                    sls._Phi_xx[tau+1] == (
                        sls._Phi_xx[tau] * sls._system_model._A  +
                        sls._Phi_xy[tau] * sls._system_model._C2
                    )
                ]

                constraints += [
                    sls._Phi_ux[tau+1] == (
                        sls._Phi_ux[tau]   * sls._system_model._A  +
                        sls._Phi_uy[tau+1] * sls._system_model._C2
                    )
                ]
        return constraints

class SLSCons_dLocalized (SLSConstraint):
    def __init__(self,
        base=None,
        actDelay=0, cSpeed=1, d=1
    ):
        '''
        actDelay: actuation delay
        cSpeed: communication speed
        d: for d-localized
        '''
        if isinstance(base,SLSCons_dLocalized):
            self._actDelay = base._actDelay
            self._cSpeed = base._cSpeed
            self._d = base._d
        else:
            self._actDelay = actDelay
            self._cSpeed = cSpeed
            self._d = d
    
    def addConstraints(self, sls, constraints):
        # localized constraints
        # get localized supports
        Phi_x = sls._Phi_x
        Phi_u = sls._Phi_u

        commsAdj = np.absolute(sls._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d - 1) > 0

        # performance helpers
        absB2T = np.absolute(sls._system_model._B2).T
        x_range_test = range(1,sls._FIR_horizon-1)

        # adjacency matrix for available information 
        infoAdj = np.eye(sls._system_model._Nx) > 0
        transmission_time = -self._cSpeed*self._actDelay
        for t in range(sls._FIR_horizon):
            transmission_time += self._cSpeed
            while transmission_time >= 1:
                transmission_time -= 1
                infoAdj = np.dot(infoAdj,commsAdj)

            support_x = np.logical_and(infoAdj, localityR)
            support_u = np.dot(absB2T,support_x) > 0

            # shutdown those not in the support
            if t in x_range_test:
                for ix,iy in np.ndindex(support_x.shape):
                    if support_x[ix,iy] == False:
                        constraints += [ Phi_x[t][ix,iy] == 0 ]
    
            for ix,iy in np.ndindex(support_u.shape):
                if support_u[ix,iy] == False:
                    constraints += [ Phi_u[t][ix,iy] == 0 ]

        return constraints

class SLSCons_Robust (SLSConstraint):
    '''
    Robust SLS (state-feedback) constraints
    '''
    def __init__(self,
        gamma_coefficient=0
    ):
        self._gamma_coefficient = gamma_coefficient

        self._gamma = None
        self._Delta = []
        self._stability_margin = -1

    def getStabilityMargin (self):
        return self._stability_margin

    def addObjectiveValue(self, sls, objective_value):
        '''
        introduce one more term: 
            gamma_coefficient * gamma
        to the objective
        '''
        self._gamma = cp.Variable(1)
        self._stability_margin = self._gamma.value

        objective_value += self._gamma_coefficient * self._gamma

        return objective_value

    def addConstraints(self, sls, constraints):
        '''
        [ zI-A, -B2 ][ Phi_x ] = I + Delta
                     [ Phi_u ]
        || Delta ||_{Epsilon_1} <= gamma
        '''
        # reset constraints
        hat_Phi_x = sls._Phi_x
        hat_Phi_u = sls._Phi_u

        Nx = sls._system_model._Nx
        self._Delta = []
        for t in range(sls._FIR_horizon+1):
            self._Delta.append(cp.Variable(shape=(Nx,Nx)))

        constraints =  [ hat_Phi_x[0] == np.eye(Nx) + self._Delta[0] ]
        constraints += [
            (sls._system_model._A  * hat_Phi_x[sls._FIR_horizon-1] +
             sls._system_model._B2 * hat_Phi_u[sls._FIR_horizon-1] +
             self._Delta[sls._FIR_horizon]
            ) == np.zeros([Nx,Nx])
        ]

        for t in range(sls._FIR_horizon-1):
            constraints += [
                self._Delta[t+1] == (
                    hat_Phi_x[t+1]
                    - sls._system_model._A  * hat_Phi_x[t]
                    - sls._system_model._B2 * hat_Phi_u[t]
                )
            ]

        # Epsilon_1 robustness
        constraints += [
            cp.norm(cp.bmat([self._Delta]),'inf') <= self._gamma
        ]

        return constraints

class SLSCons_ApproxdLocalized (SLSCons_dLocalized, SLSCons_Robust):
    def __init__(self,
        robCoeff=0,
        **kwargs
    ):
        SLSCons_dLocalized.__init__(self,**kwargs)

        base = kwargs.get('base')
        if isinstance(base,SLSCons_ApproxdLocalized):
            self._robCoeff = base._robCoeff
        else:
            self._robCoeff = robCoeff

        SLSCons_Robust.__init__(self,gamma_coefficient=robCoeff)

    def addObjectiveValue(self, sls, objective_value):
        return SLSCons_Robust.addObjectiveValue(self, sls, objective_value)

    def addConstraints(self, sls, constraints):
        constraints = SLSCons_Robust.addConstraints(self, sls, constraints)
        constraints = SLSCons_dLocalized.addConstraints(self, sls, constraints)

        return constraints