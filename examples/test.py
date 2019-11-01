from slspy.SystemModel import LTISystem
from slspy.Simulator import Simulator
from slspy.SynthesisAlgorithm import *
from slspy.NoiseModel import *
from slspy.PlantGenerator import *
from slspy.VisualizationTools import *
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