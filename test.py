from sls_sim.SystemModel import LTISystem
import numpy as np

def test ():
    model = LTISystem()
    model._A = np.empty ([2,2])
    model._B1 = np.empty ([2,2])
    model._B2 = np.empty ([2,0])
    
    model.stateFeedback(False)
    model.sanityCheck()

if __name__ == '__main__':
    test ()