from SystemModel import SystemModel

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
            # using the horizon specified 
        return x, u