from .Base import ObjBase
from .SystemModel import *
from .ControllerModel import *
from .SLSHelpers import *
from enum import Enum, unique
import cvxpy as cp

class SynthesisAlgorithm (ObjBase):
    '''
    The base class for synthesis algorithm, which takes a system model and generates a controller model correspondingly.
    '''
    def __init__(self,system_model=None):
        self.setSystemModel(system_model=system_model)

    def setSystemModel(self,system_model):
        if isinstance(system_model,SystemModel):
            self._system_model = system_model
    
    def synthesizeControllerModel(self):
        return None

class SLS (SynthesisAlgorithm):
    '''
    Synthesizing the controller using System Level Synthesis method.
    '''
    @unique
    class Objective (Enum):
        # specify the number for the support og python 2.7
        ZERO = 0
        H2 = 1
        HInf = 2
        L1 = 3

    def __init__(self,
        base=None,
        system_model=None,
        FIR_horizon=1,
        state_feedback=True,
        obj_type=Objective.ZERO
    ):
        if isinstance(base,SLS):
            self.setSystemModel(system_model=base._system_model)
            self._FIR_horizon = base._FIR_horizon
            self._state_feedback = base._state_feedback
            self.setObjectiveType(base._obj_type)
        else:
            self.setSystemModel(system_model=system_model)
            self._FIR_horizon = FIR_horizon
            self._state_feedback = state_feedback
            self.setObjectiveType(obj_type)

        self._optimal_objective_value = float('inf')
    
    def setObjectiveType(self,obj_type=Objective.ZERO):
        if isinstance(obj_type,self.Objective):
            self._obj_type=obj_type
        else:
            self._obj_type=Objective.ZERO
    
    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value.copy()

    def sanityCheck (self):
        # TODO: we can extend the algorithm to work for non-state-feedback SLS
        if not self._state_feedback:
            return self.errorMessage('Only support state-feedback case for now.')

        if not isinstance(self._system_model,LTISystem):
            return self.errorMessage('The system must be LTI.')
        if not isinstance(self._FIR_horizon,int):
            return self.errorMessage('FIR horizon must be integer.')
        if self._FIR_horizon < 1:
            return self.errorMessage('FIR horizon must be at least 1.')

        return True
    
    def synthesizeControllerModel(self):
        self._optimal_objective_value = float('inf')
        if not self.sanityCheck():
            return None
        if not self._system_model.sanityCheck():
            self.errorMessage('System model check fails.')
            return None

        if self._state_feedback:
            Nx = self._system_model._Nx
            Nu = self._system_model._Nu
            controller = SLS_State_Feedback_FIR_Controller(
                Nx=Nx, Nu=Nu,
                FIR_horizon=self._FIR_horizon
            )

            # declare variables
            Phi_x = []
            Phi_u = []
            for tau in range(self._FIR_horizon):
                Phi_x.append(cp.Variable(shape=(Nx,Nx)))
                Phi_u.append(cp.Variable(shape=(Nu,Nx)))

            # objective
            objective_value = self.__getObjectiveValue(Phi_x=Phi_x,Phi_u=Phi_u)
            if objective_value is None:
                self.errorMessage('Objective generation fails.')
                return None

            # sls constraints
            constraints = [ Phi_x[0] == np.eye(Nx) ]
            constraints += [ Phi_x[self._FIR_horizon-1] == np.zeros([Nx, Nx]) ]
            for tau in range(self._FIR_horizon-1):
                constraints += [
                    Phi_x[tau+1] == (
                        self._system_model._A  * Phi_x[tau] +
                        self._system_model._B2 * Phi_u[tau]
                    )
                ]

            # additional constraints
            objective_value, constraints = self._additionalObjectiveOrConstraints(
                Phi_x=Phi_x,
                Phi_u=Phi_u,
                objective_value=objective_value,
                constraints=constraints
            )

            # obtain results and put into controller
            sls_problem = cp.Problem(cp.Minimize(objective_value),constraints)
            sls_problem.solve()

            if sls_problem.status is "infeasible":
                self.warningMessage('SLS problem infeasible')
                return None
            elif sls_problem.status is "unbounded":
                self.warningMessage('SLS problem unbounded')
                return None
            else:
                self._optimal_objective_value = sls_problem.value
                controller._Phi_x = []
                controller._Phi_u = []
                for tau in range(self._FIR_horizon):
                    controller._Phi_x.append(Phi_x[tau].value)
                    controller._Phi_u.append(Phi_u[tau].value)
                controller.initialize()

            return controller
        else:
            # TODO
            self.errorMessage('Not yet support the output-feedback case.')
            return None
    
    def __getObjectiveValue(self,Phi_x,Phi_u):
        objective_value = None

        if self._obj_type != SLS.Objective.ZERO:
            if self._system_model._ignore_output:
                self.warningMessage('H2 output ignored. Objective is zero.')
                objective_value = 0
            else:
                obj_val_function = None
                if self._obj_type == SLS.Objective.H2:
                    obj_val_function = SLS_Objective_Value_H2
                elif self._obj_type == SLS.Objective.HInf:
                    obj_val_function = SLS_Objective_Value_HInf
                elif self._obj_type == SLS.Objective.L1:
                    obj_val_function = SLS_Objective_Value_L1
                
                objective_value = obj_val_function (
                    C1=self._system_model._C1,
                    D12=self._system_model._D12,
                    Phi_x=Phi_x,
                    Phi_u=Phi_u
                )
        else:  # self._obj_type == SLS.Objective.ZERO:
            self.warningMessage('Objective is zero.')
            objective_value = 0
        
        # we can extend the function here to include some penalty function
        return objective_value
    
    def _additionalObjectiveOrConstraints(self,Phi_x=[],Phi_u=[],objective_value=None, constraints=None):
        # for inherited classes to introduce additional terms
        return objective_value, constraints

class dLocalizedSLS (SLS):
    def __init__(self,
        actDelay=0, cSpeed=1, d=1,
        **kwargs
    ):
        SLS.__init__(self,**kwargs)

        base = kwargs.get('base')
        if isinstance(base,dLocalizedSLS):
            self._actDelay = base._actDelay
            self._cSpeed = base._cSpeed
            self._d = base._d
        else:
            self._actDelay = actDelay
            self._cSpeed = cSpeed
            self._d = d
    
    def _additionalObjectiveOrConstraints(self,Phi_x=[],Phi_u=[],objective_value=None, constraints=None):
        # localized constraints
        # get localized supports
        XSupport = []
        USupport = []

        commsAdj = np.absolute(self._system_model._A) > 0
        localityR = np.linalg.matrix_power(commsAdj, self._d - 1) > 0

        # adjacency matrix for available information 
        infoAdj = np.eye(self._system_model._Nx) > 0
        transmission_time = -self._cSpeed*self._actDelay
        for t in range(self._FIR_horizon):
            transmission_time += self._cSpeed
            while transmission_time >= 1:
                transmission_time -= 1
                infoAdj = np.dot(infoAdj,commsAdj)

            support_x = np.logical_and(infoAdj, localityR)
            XSupport.append(support_x)

            support_u = np.dot(np.absolute(self._system_model._B2).T,support_x.astype(int)) > 0
            USupport.append(support_u)

        # shutdown those not in the support
        for t in range(1,self._FIR_horizon-1):
            for ix,iy in np.ndindex(XSupport[t].shape):
                if XSupport[t][ix,iy] == False:
                    constraints += [ Phi_x[t][ix,iy] == 0 ]
        for t in range(self._FIR_horizon):
            for ix,iy in np.ndindex(USupport[t].shape):
                if USupport[t][ix,iy] == False:
                    constraints += [ Phi_u[t][ix,iy] == 0 ]

        return objective_value, constraints

class ApproxdLocalizedSLS (dLocalizedSLS):
    def __init__(self,
        robCoeff=0,
        **kwargs
    ):
        dLocalizedSLS.__init__(self,**kwargs)

        base = kwargs.get('base')
        if isinstance(base,ApproxdLocalizedSLS):
            self._robCoeff = base._robCoeff
        else:
            self._robCoeff = robCoeff

        self._stability_margin = -1

    def getStabilityMargin (self):
        return self._stability_margin

    def _additionalObjectiveOrConstraints(self,Phi_x=[],Phi_u=[],objective_value=None, constraints=None):
        # reset constraints
        Nx = self._system_model._Nx
        constraints = [ Phi_x[0] == np.eye(Nx) ]
        constraints += [ Phi_x[self._FIR_horizon-1] == np.zeros([Nx, Nx]) ]

        dLocalizedSLS._additionalObjectiveOrConstraints(self,
            Phi_x=Phi_x,
            Phi_u=Phi_u,
            objective_value=objective_value,
            constraints=constraints
        )

        Delta = cp.Variable(shape=(Nx,Nx*self._FIR_horizon))

        pos = 0
        for t in range(self._FIR_horizon-1):
            constraints += [
                Delta[:,pos:pos+Nx] == (
                    Phi_x[t+1]
                    - self._system_model._A  * Phi_x[t]
                    - self._system_model._B2 * Phi_u[t]
                )
            ]
            pos += Nx

        self._stability_margin = cp.norm(Delta, 'inf')  # < 1 means we can guarantee stability
        objective_value += self._robCoeff * self._stability_margin

        return objective_value, constraints