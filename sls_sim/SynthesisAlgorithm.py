from .Base import ObjBase
from .SystemModel import *
from .ControllerModel import *
from .SLSObjective import SLSObjective
from .SLSConstraint import SLSConstraint
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
    def __init__(self,
        system_model=None,
        FIR_horizon=1,
        state_feedback=True
    ):
        self.setSystemModel(system_model=system_model)
        self._FIR_horizon = FIR_horizon
        self._state_feedback = state_feedback
        
        self.resetObjAndCons()
    
    # overload plus operator
    def __add__(self, obj_or_cons):
        if isinstance(obj_or_cons, SLSObjective):
            self._objectives.append(obj_or_cons)
        if isinstance(obj_or_cons, SLSConstraint):
            self._constraints.append(obj_or_cons)
        return self

    def resetObjAndCons (self):
        self.resetObjectives ()
        self.resetConstraints ()
    
    def resetObjectives (self):
        self._objectives = []
        self._optimal_objective_value = float('inf')
    
    def resetConstraints (self):
        self._constraints = []

    def setObjOrCons (self, obj_or_cons):
        if isinstance(obj_or_cons, SLSObjective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        if isinstance(obj_or_cons, SLSConstraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)

    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value.copy()

    def sanityCheck (self):
        # TODO: we can extend the algorithm to work for non-state-feedback SLS
        if not self._state_feedback:
            return self.errorMessage('Only support state-feedback case for now.')

        if self._system_model is None:
            return self.errorMessage('The system is not yet assigned.')
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
            self._Phi_x = []
            self._Phi_u = []
            for tau in range(self._FIR_horizon):
                self._Phi_x.append(cp.Variable(shape=(Nx,Nx)))
                self._Phi_u.append(cp.Variable(shape=(Nu,Nx)))

            # objective
            objective_value = 0
            for obj in self._objectives:
                objective_value = obj.addObjectiveValue (
                    sls = self,
                    objective_value = objective_value
                )

            # sls constraints
            constraints =  [ self._Phi_x[0] == np.eye(Nx) ]
            constraints += [ self._Phi_x[self._FIR_horizon-1] == np.zeros([Nx, Nx]) ]
            for tau in range(self._FIR_horizon-1):
                constraints += [
                     self._Phi_x[tau+1] == (
                        self._system_model._A  * self._Phi_x[tau] +
                        self._system_model._B2 * self._Phi_u[tau]
                    )
                ]

            # the constraints might also introduce additional terms at the objective
            for cons in self._constraints:
                objective_value, constraints = cons.addConstraints (
                    sls = self,
                    objective_value = objective_value,
                    constraints = constraints
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
                    controller._Phi_x.append(self._Phi_x[tau].value)
                    controller._Phi_u.append(self._Phi_u[tau].value)
                controller.initialize()

            return controller
        else:
            # TODO
            self.errorMessage('Not yet support the output-feedback case.')
            return None