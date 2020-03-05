import inspect
import numpy as np
'''
The core abstract classes that form the SLSpy framework:

    SystemModel -> SynthesisAlgorithm -> ControllerModel

    SystemModel, ControllerModel, NoiseModel -> Simulator -> simulation results
'''

class ObjBase:
    '''
    The object base that defines debugging tools
    '''
    def initialize (self, **kwargs):
        pass

    def sanityCheck (self):
        # check the system parameters are coherent
        return True

    def errorMessage (self,msg):
        print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [ERROR] '+msg+'\n')
        return False

    def warningMessage (self,msg):
        print(self.__class__.__name__+'-'+inspect.stack()[1][3]+': [WARNING] '+msg+'\n')
        return False

class SystemModel (ObjBase):
    '''
    The base class for discrete-time system models.
    A controller synthesizer takes a system model and synthesizes a controller.
    '''
    def __init__ (self,
        ignore_output=False,
        state_feedback=True
    ):
        self._x = np.empty([0])  # state
        self._y = np.empty([0])  # measurement
        self._z = np.empty([0])  # regularized output

        self._ignore_output = ignore_output
        self._state_feedback = state_feedback

        self._x0 = None

    def systemProgress (self, u, w=None, **kwargs):
        # this function takes the input (and the noise) and progress to next time 
        pass

    def getState(self):
        return self._x.copy()

    def getMeasurement(self):
        if self._state_feedback:
            return self._x.copy()
        else:
            return self._y.copy()
    
    def getOutput(self):
        if self._ignore_output:
            return None
        else:
            return self._z.copy()

    def ignoreOutput (self, ignore_output=False):
        # Does this model ignore output?
        self._ignore_output = ignore_output

    def stateFeedback (self, state_feedback=True):
        # Is this model state-feedback?
        self._state_feedback = state_feedback

class ControllerModel (ObjBase):
    '''
    The base class for discrete-time controller.
    '''
    def initialize (self):
        # initialize internal state
        pass

    def getControl(self, y, **kwargs):
        return None

class NoiseModel (ObjBase):
    '''
    The base class for noise model.
    NoiseModel is responsible for the right format of noise (dimension, etc.)
    '''
    def __init__ (self, Nw=0):
        self._Nw = Nw  # dimension of the noise (disturbance)

    def initialize (self):
        # auto-initialization for each simulation
        pass

    def getNoise (self,**kwargs):
        # the noise can depend on some parameters such as state or control
        return 0

class SynthesisAlgorithm (ObjBase):
    '''
    The base class for synthesis algorithm, which takes a system model and generates a controller model correspondingly.
    '''
    def __init__(self,system_model=None):
        self.setSystemModel(system_model=system_model)

    # overload the less than or equal operator as a syntactic sugar
    def __lshift__ (self, sytem):
        return self.setSystemModel(system_model=system)

    def setSystemModel(self,system_model):
        if isinstance(system_model,SystemModel):
            self._system_model = system_model
        return self
    
    def synthesizeControllerModel(self):
        return None

class Simulator (ObjBase):
    '''
    The simulator
    '''
    def __init__ (self, 
        system=None,
        controller=None,
        noise=None,
        horizon=-1
    ):
        self.setSystem (system)
        self.setController (controller)
        self.setNoise (noise)
        self.setHorizon (horizon)
        pass

    def setSystem (self, system=None):
        if isinstance(system, SystemModel):
            self._system = system

    def setController (self, controller=None):
        if isinstance(controller, ControllerModel):
            self._controller = controller

    def setNoise (self, noise=None):
        if isinstance(noise, NoiseModel):
            self._noise = noise
        else:
            self._noise = None

    def setHorizon (self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def run (self,initialize=True):
        # run the system and return 
        #   system state (x)
        #   system measurement (y)
        #   system output (z)
        #   control history (u)
        #   noise history (w)
        if self._horizon < 0:
            return None, None, None, None

        if not self._system.sanityCheck ():
            return None, None, None, None

        if initialize:
            # initialize
            self._system.initialize()
            self._controller.initialize()
            if self._noise is not None:
                self._noise.initialize()

        x_history = []
        y_history = []
        z_history = []
        u_history = []
        w_history = []

        for t in range (self._horizon):
            x = self._system.getState()
            x_history.append(x)
            y = self._system.getMeasurement()
            y_history.append(y)
            z = self._system.getOutput()
            z_history.append(z)

            u = self._controller.getControl(y=y)
            u_history.append(u)
            if self._noise is not None:
                w = self._noise.getNoise()
            else:
                w = None
            w_history.append(w)
            self._system.systemProgress(u=u, w=w)

        return x_history, y_history, z_history, u_history, w_history