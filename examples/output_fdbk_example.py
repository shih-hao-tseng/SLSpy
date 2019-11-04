from slspy.system_model import LTISystem
from slspy.controller_model import *
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import SLS
from slspy.sls_objective import *
from slspy.sls_constraint import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
import numpy as np

def output_fdbk_example():
    sys = LTISystem (
        Nx = 10, Ny = 10, Nw = 10,
        state_feedback = False
    )

    # generate sys._A, sys._B2
    generate_doubly_stochastic_chain (
        system_model = sys,
        rho = 1,
        actuator_density = 1,
        alpha = 0.2
    )
    generate_BCD_and_zero_initialization(sys)

    sim_horizon = 25
    simulator = Simulator (
        system = sys,
        horizon = sim_horizon
    )

    # generate noise
    noise = FixedNoiseVector (Nw = sys._Nx, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nx/2] = 10

    sys.useNoiseModel (noise_model = noise)

    # use SLS controller synthesis algorithm
    # notice that the system should also be output-feedback (state_feedback = False)
    # to actually trigger the output-feedback SLS
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 20,
        state_feedback = False # use output-feedback synthesizer
    )

    # set SLS objective
    synthesizer <= SLSObj_H2()

    # synthesize controller (the generated controller is actually initialized)
    # and use the synthesized controller in simulation
    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (
        controller = controller
    )

    noise.startAtTime(0)

    # run the simulation
    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Centralized')

if __name__ == '__main__':
    output_fdbk_example()