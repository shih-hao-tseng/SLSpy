from .base import ObjBase
from .system_model import SystemModel
from .controller_model import ControllerModel
import numpy as np

class Simulator (ObjBase):
    '''
    The simulator
    '''
    def __init__ (self, system=None, controller=None, horizon=-1):
        self.setSystem (system)
        self.setController (controller)
        self.setHorizon (horizon)
        pass

    def setSystem (self, system=None):
        if isinstance(system, SystemModel):
            self._system = system

    def setController (self, controller=None):
        if isinstance(controller, ControllerModel):
            self._controller = controller
    
    def setHorizon (self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def run (self):
        # run the system and return 
        #   system state (x)
        #   system measurement (y)
        #   system output (z)
        #   control history (u)
        if self._horizon < 0:
            return None, None, None, None

        if not self._system.sanityCheck ():
            return None, None, None, None

        x_history = []
        y_history = []
        z_history = []
        u_history = []

        for t in range (self._horizon):
            x = self._system.getState()
            x_history.append(x)
            y = self._system.getMeasurement()
            y_history.append(y)
            z = self._system.getOutput()
            z_history.append(z)

            u = self._controller.getControl(y=y)
            u_history.append(u)
            self._system.systemProgress(u=u)

        return x_history, y_history, z_history, u_history