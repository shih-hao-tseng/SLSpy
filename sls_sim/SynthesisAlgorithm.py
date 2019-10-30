from Base import ObjBase
from SystemModel import *
from ControllerModel import *

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
    def __init__(self,system_model=None,FIR_horizon=1,state_feedback=True):
        self.setSystemModel(system_model=system_model)
        self._FIR_horizon = FIR_horizon
        self._state_feedback = state_feedback
    
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
        self.sanityCheck()

        if self._state_feedback:
            controller = SLS_State_Feedback_FIR_Controller(
                Nx=self._system_model._Nx,
                Nu=self._system_model._Nu,
                FIR_horizon=self._FIR_horizon
            )

            # TODO: the algorithm




            return controller
        else:
            return None