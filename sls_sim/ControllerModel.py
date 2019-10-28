class ControllerModel (ObjBase):
    '''
    The base class for discrete-time controller.
    '''
    def getControl(self, measurement, **kwargs):
        return None