from SystemModel import SystemModel
from ControllerModel import ControllerModel

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
        if isinstance(system, ControllerModel):
            self._controller = controller
    
    def setHorizon (self, horizon=-1):
        if isinstance(horizon, int):
            self._horizon = horizon

    def run (self):
        if self._horizon < 0:
            return None, None

        if not self.system.sanityCheck ():
            return None, None

        y_history = []
        u_history = []

        for t in range (self._horizon):
            y = self.system.getMeasurement()
            y_history.append(y)
            u = self.controller.getControl(y=y)
            u_history.append(u)
            self.system.systemProgress(u=u)

        return y_history, u_history