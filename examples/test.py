from slspy.system_model import LTISystem
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
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