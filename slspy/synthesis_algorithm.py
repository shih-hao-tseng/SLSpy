from .base import ObjBase
from .system_model import *
from .controller_model import *
from .sls_objective import SLSObjective
from .sls_constraint import SLSConstraint, SLSCons_SLS
import cvxpy as cp

class SynthesisAlgorithm (ObjBase):
    '''
    The base class for synthesis algorithm, which takes a system model and generates a controller model correspondingly.
    '''
    def __init__(self,system_model=None):
        self.setSystemModel(system_model=system_model)

    # overload the less than or equal operator as a syntactic sugar
    def __le__ (self, sytem):
        return self.setSystemModel(system_model=system)

    def setSystemModel(self,system_model):
        if isinstance(system_model,SystemModel):
            self._system_model = system_model
        return self
    
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

        self._Phi_x = self._Phi_xx = []
        self._Phi_u = self._Phi_ux = []
        self._Phi_xy = []
        self._Phi_uy = []

        self._sls_problem = None
        self._sls_constraints = SLSCons_SLS ()
   
    # overload plus and less than or equal operators as syntactic sugars
    def __add__(self, obj_or_cons):
        return self.addObjOrCons(obj_or_cons)

    def __le__ (self, obj_or_cons_or_system):
        if isinstance(obj_or_cons_or_system,SystemModel):
            return self.setSystemModel(system_model=obj_or_cons_or_system)
        else:
            return self.setObjOrCons(obj_or_cons=obj_or_cons_or_system)

    def resetObjAndCons (self):
        self.resetObjectives ()
        self.resetConstraints ()
    
    def resetObjectives (self):
        self._objectives = []
        self._optimal_objective_value = float('inf')
    
    def resetConstraints (self):
        self._constraints = []

    def addObjOrCons (self, obj_or_cons):
        if isinstance(obj_or_cons, SLSConstraint):
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLSObjective):
            self._objectives.append(obj_or_cons)
        return self

    def setObjOrCons (self, obj_or_cons):
        if isinstance(obj_or_cons, SLSConstraint):
            self._constraints = []
            self._constraints.append(obj_or_cons)
        elif isinstance(obj_or_cons, SLSObjective):
            self._objectives = []
            self._objectives.append(obj_or_cons)
        return self

    def getOptimalObjectiveValue (self):
        return self._optimal_objective_value.copy()

    def getSLSProblem (self):
        return self._sls_problem

    def sanityCheck (self):
        # we can extend the algorithm to work for non-state-feedback SLS
        #if not self._state_feedback:
        #    return self.errorMessage('Only support state-feedback case for now.')

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

        # variables used by both the state-feedback and output-feedback versions
        Nx = self._system_model._Nx
        Nu = self._system_model._Nu

        self._Phi_xx = self._Phi_x = []
        self._Phi_ux = self._Phi_u = []

        use_state_feedback_version = self._state_feedback or self._system_model._state_feedback

        if use_state_feedback_version:
            controller = SLS_State_Feedback_FIR_Controller (
                Nx=Nx, Nu=Nu,
                FIR_horizon=self._FIR_horizon
            )

            # declare variables
            for tau in range(self._FIR_horizon):
                self._Phi_x.append(cp.Variable(shape=(Nx,Nx)))
                self._Phi_u.append(cp.Variable(shape=(Nu,Nx)))
        else:
            # output-feedback
            Ny = self._system_model._Ny

            controller = SLS_Output_Feedback_FIR_Controller (
                Nx=Nx, Nu=Nu, Ny=Ny, D22=self._system_model._D22,
                FIR_horizon=self._FIR_horizon
            )

            # declare variables
            self._Phi_xy = []
            self._Phi_uy = []
            for tau in range(self._FIR_horizon):
                self._Phi_xx.append(cp.Variable(shape=(Nx,Nx)))
                self._Phi_ux.append(cp.Variable(shape=(Nu,Nx)))
                self._Phi_xy.append(cp.Variable(shape=(Nx,Ny)))
                self._Phi_uy.append(cp.Variable(shape=(Nu,Ny)))
            # Phi_uy is in RH_{\inf} instead of z^{-1} RH_{\inf}
            self._Phi_uy.append(cp.Variable(shape=(Nu,Ny)))

        # objective
        objective_value = 0
        for obj in self._objectives:
            objective_value = obj.addObjectiveValue (
                sls = self,
                objective_value = objective_value
            )

        # add SLS main constraints
        self._sls_constraints._state_feedback = use_state_feedback_version
        constraints = self._sls_constraints.addConstraints (sls = self)

        # the constraints might also introduce additional terms at the objective
        for cons in self._constraints:
            objective_value = cons.addObjectiveValue (
                sls = self,
                objective_value = objective_value
            )
            constraints = cons.addConstraints (
                sls = self,
                constraints = constraints
            )

        # obtain results and put into controller
        sls_problem = cp.Problem(cp.Minimize(objective_value),constraints)
        sls_problem.solve()

        self._sls_problem = None

        if sls_problem.status is "infeasible":
            self.warningMessage('SLS problem infeasible')
            return None
        elif sls_problem.status is "unbounded":
            self.warningMessage('SLS problem unbounded')
            return None
        else:
            # save the solved problem for the users to examine if needed
            self._sls_problem = sls_problem
            self._optimal_objective_value = sls_problem.value
            if use_state_feedback_version:
                controller._Phi_x = []
                controller._Phi_u = []
                for tau in range(self._FIR_horizon):
                    controller._Phi_x.append(self._Phi_x[tau].value)
                    controller._Phi_u.append(self._Phi_u[tau].value)
            else:
                controller._Phi_xx = []
                controller._Phi_ux = []
                controller._Phi_xy = []
                controller._Phi_uy = []
                for tau in range(self._FIR_horizon):
                    controller._Phi_xx.append(self._Phi_xx[tau].value)
                    controller._Phi_ux.append(self._Phi_ux[tau].value)
                    controller._Phi_xy.append(self._Phi_xy[tau].value)
                    controller._Phi_uy.append(self._Phi_uy[tau].value)

        controller.initialize()
        return controller