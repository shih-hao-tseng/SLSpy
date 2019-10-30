from sls_sim.SystemModel import LTISystem
from sls_sim.NoiseModel import *
import numpy as np

def test ():
    model = LTISystem(
        Nx=2, Nw=2, Nu=1
    )
    model._B1 = np.eye (2)
    
    model.stateFeedback()
    model.ignoreOutput()

    fixed_noise = FixedNoiseVector (horizon=10)
    fixed_noise.generateNoiseFromNoiseModelInstance (noise_model=model._noise_model)
    fixed_noise.startAtTime(0)

    w = fixed_noise.getNoise()
    print (w)
    w = fixed_noise.getNoise()
    print (w)
    fixed_noise.startAtTime(0)
    w = fixed_noise.getNoise()
    print (w)
    fixed_noise.startAtTime(0)

    model.useNoiseModel(noise_model=fixed_noise)
    model.initialize(x0=np.zeros((2,1)))

    model.sanityCheck()

    print(model.getState())

    model.systemProgress(u=np.zeros([1,1]))

    print(model.getState())

    sp = np.concatenate( (np.eye(2),np.zeros([2,3])), axis=1) 
    print(sp)

if __name__ == '__main__':
    test ()