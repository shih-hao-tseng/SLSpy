from Base import ObjBase
import numpy as np

class NoiseModel (ObjBase):
    '''
    The base class for noise model.
    NoiseModel is responsible for the right format of noise (dimension, etc.)
    '''
    def __init__ (self):
        self._Nw = 0  # dimension of the noise (disturbance)

    def getNoise (self,**kwargs):
        # the noise can depend on some parameters such as state or control
        pass
