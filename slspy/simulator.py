from .base import ObjBase
from .system_model import SystemModel
from .controller_model import ControllerModel
from .noise_model import NoiseModel
import numpy as np

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