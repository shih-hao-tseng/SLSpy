from .core import Noise_Model
import numpy as np
'''
To create a new noise model, inherit the following base function and customize the specified methods.

class NoiseModel:
    def __init__ (self, Nw=0):
    def initialize (self):
    def getNoise (self,**kwargs):
        # the noise can depend on some parameters such as state or control
        return w
'''

class Zero_Noise (Noise_Model):
    '''
    Generate zero vector as the noise
    '''
    def __init__ (self, Nw=0):
        self.setDimension(Nw)
    
    def setDimension(self, Nw=0):
        self._Nw = Nw
        self._w = np.zeros([Nw,1])

    def getNoise(self,**kwargs):
        return self._w.copy()


class Gaussian_Noise(Noise_Model):
    '''
    Generate Gaussian noise
    '''
    def __init__ (self, Nw=0, mu=0, sigma=1):
        Noise_Model.__init__(self,Nw=Nw)

        self._mu = mu
        self._sigma = sigma
    
    def getNoise (self,**kwargs):
        return np.random.normal (self._mu, self._sigma, (self._Nw,1))
    
class Fixed_Noise_Vector(Noise_Model):
    '''
    Fixed noise vector
    '''
    def __init__ (self, Nw=0, horizon=0, t0=0):
        Noise_Model.__init__(self,Nw=Nw)
        self._horizon = horizon
        self._t = 0
        self._t0 = 0
        self._w = []

    def initialize (self):
        self.startAtTime(t=self._t0)

    def startAtTime(self, t=0):
        self._t = self._t0 = t

    def generateNoiseFromNoiseModelInstance (self, noise_model=None):
        if not isinstance (noise_model, Noise_Model):
            return

        self._Nw = noise_model._Nw

        self._w = []
        for t in range (self._horizon):
            self._w.append(noise_model.getNoise())
        
    def generateNoiseFromNoiseModel (self, cls=Noise_Model):
        noise_model = cls(Nw=self._Nw)
        self.generateNoiseFromNoiseModelInstance (noise_model=noise_model)
    
    def setNoise (self,w=None):
        # directly assign the _w vector
        self._w = w

    def getNoise (self,**kwargs):
        if self._t < self._horizon:
            w = self._w[self._t]
            self._t += 1
            return w

        return np.zeros((self._Nw,1))