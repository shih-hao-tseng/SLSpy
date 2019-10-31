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

    def __init__(self,system_model=None,FIR_horizon=1,state_feedback=True,obj_type=Objective.ZERO):
        self.setSystemModel(system_model=system_model)
        self._FIR_horizon = FIR_horizon
        self._state_feedback = state_feedback
        self._optimal_objective_value = float('inf')
        self.setObjectiveType(obj_type)
    
    def setObjectiveType(self,obj_type=Objective.ZERO):
        if isinstance(obj_type,self.Objective):
            self._obj_type=obj_type
        else:
            self._obj_type=Objective.ZERO
    
    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value

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
            objective = self.__getObjective(Phi_x=Phi_x,Phi_u=Phi_u)
            if objective is None:
                self.errorMessage('Objective generation fails.')
                return None

            # sls constraints
            constraints = [ Phi_x[0] == np.eye(Nx) ]
            constraints += [ Phi_x[self._FIR_horizon-1] == np.zeros([Nx, Nx]) ]
            for tau in range(self._FIR_horizon-1):
                constraints += [ Phi_x[tau+1] == (
                        self._system_model._A * Phi_x[tau] +
                        self._system_model._B2 * Phi_u[tau]
                    )
                ]

            # obtain results and put into controller
            sls_problem = cp.Problem(objective,constraints)
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
    
    def __getObjective(self,Phi_x,Phi_u):
        objective_value = None

        if self._obj_type == SLS.Objective.H2:
            if self._system_model._ignore_output:
                self.warningMessage('H2 output ignored. Objective is zero.')
                objective_value = 0
            else:
                objective_value = SLS_Objective_Value_H2(
                    C1=self._system_model._C1,
                    D12=self._system_model._D12,
                    Phi_x=Phi_x,
                    Phi_u=Phi_u
                )
        elif self._obj_type == SLS.Objective.HInf:
            # TODO
            pass
        elif self._obj_type == SLS.Objective.L1:
            # TODO
            pass
        else:  # self._obj_type == SLS.Objective.ZERO:
            self.warningMessage('Objective is zero.')
            objective_value = 0
        
        # we can extend the function here to include some penalty function
        return cp.Minimize(objective_value)