from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import *
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
from sls_sim.VisualizationTools import *
import numpy as np

def test():
    sys = LTISystem (
        Nx = 10, Nw = 10
    )
    generate_random_chain(
        system_model = sys,
        rho = 0.5,
        actuator_density = 0.3
    )
    sys.ignoreOutput(True)
    sys.sanityCheck()

if __name__ == '__main__':
    test()