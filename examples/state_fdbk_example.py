from slspy.system_model import LTISystem
from slspy.simulator import Simulator
from slspy.synthesis_algorithm import *
from slspy.sls_objective import *
from slspy.sls_constraint import *
from slspy.noise_model import *
from slspy.plant_generator import *
from slspy.visualization_tool import *
import numpy as np

def state_fdbk_example():
    sys = LTISystem (
        Nx = 10, Nw = 10
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


    ## (1) basic sls (centralized controller)
    # use SLS controller synthesis algorithm
    synthesizer = SLS (FIR_horizon = 20)
    synthesizer += SLSObj_H2()
    synthesizer.setSystemModel (sys)

    # synthesize controller (the generated controller is actually initialized)
    controller = synthesizer.synthesizeControllerModel ()

    # use the synthesized controller in simulation
    simulator.setController (controller=controller)

    # initialize the system and the controller
    sys.initialize ()
    controller.initialize ()
    noise.startAtTime(0)

    # run the simulation
    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Centralized')


    ## (2) d-localized sls
    dlocalized = SLSCons_dLocalized (
        actDelay = 1,
        cSpeed = 2,
        d = 3
    )
    synthesizer += dlocalized

    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (controller=controller)

    # reuse the predefined initialization
    sys.initialize ()
    controller.initialize ()
    noise.startAtTime(0)

    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Localized')


    ## (3) approximate d-localized sls
    approx_dlocalized = SLSCons_ApproxdLocalized (
        base = dlocalized,
        robCoeff = 10e3
    )
    approx_dlocalized._cSpeed = 1

    # set the constriant
    synthesizer.setObjOrCons(approx_dlocalized)

    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (controller=controller)

    # reuse the predefined initialization
    sys.initialize ()
    controller.initialize ()
    noise.startAtTime(0)

    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Approximately Localized')


if __name__ == '__main__':
    state_fdbk_example()