from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import *
from sls_sim.SLSObjective import *
from sls_sim.SLSConstraint import *
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
from sls_sim.VisualizationTools import *
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

    # specify system matrices
    sys._B1  = np.eye (sys._Nx)
    sys._C1  = np.concatenate ((np.eye(sys._Nx), np.zeros([sys._Nu, sys._Nx])), axis = 0)
    sys._D12 = np.concatenate ((np.zeros([sys._Nx, sys._Nu]), np.eye(sys._Nu)), axis = 0)
    sys.initialize (x0 = np.zeros([sys._Nx, 1]))

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
    # add d-localized constriants
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