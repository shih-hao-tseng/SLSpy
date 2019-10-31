from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import *
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
from sls_sim.VisualizationTools import *
import numpy as np

def state_fdbk_example():
    sys = LTISystem (
        Nx = 10, Nw = 10
    )

    # generate sys._A, sys._B2
    GenerateDoublyStochasticChain (
        system_model = sys,
        rho = 1,
        actuator_density = 1,
        alpha = 0.2
    )

    # specify system matrices
    sys._B1  = np.eye (sys._Nx)
    sys._C1  = np.concatenate ((np.eye(sys._Nx), np.zeros([sys._Nu, sys._Nx])), axis = 0)
    sys._D12 = np.concatenate ((np.zeros([sys._Nx, sys._Nu]), np.eye(sys._Nu)), axis = 0)

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
    synthesizer = SLS (
        FIR_horizon = 20,
        obj_type = SLS.Objective.H2
    )
    synthesizer.setSystemModel (sys)

    # synthesize controller (the generated controller is actually initialized)
    controller = synthesizer.synthesizeControllerModel ()

    # use the synthesized controller in simulation
    simulator.setController (controller=controller)

    # initialize the system and the controller
    sys.initialize (x0 = np.zeros([sys._Nx, 1]))
    controller.initialize ()

    # run the simulation
    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = Matrix_List_Multiplication(sys._B2,u_history)
    Plot_Heat_Map(x_history, Bu_history, 'Centralized')

    ## (2) d-localized sls
    dlocalized_synthesizer = dLocalizedSLS (
        base = synthesizer,
        actDelay = 1,
        cSpeed = 2,
        d = 3
    )
    controller = dlocalized_synthesizer.synthesizeControllerModel ()
    simulator.setController (controller=controller)

    # reuse the predefined initialization
    sys.initialize ()
    controller.initialize ()

    x_history, y_history, z_history, u_history = simulator.run ()

    Bu_history = Matrix_List_Multiplication(sys._B2,u_history)
    Plot_Heat_Map(x_history, Bu_history, 'Localized')

#    slsParams.mode_     = SLSMode.DLocalized
#    slsParams.actDelay_ = 1
#    slsParams.cSpeed_   = 2 # communication speed must be sufficiently large
#    slsParams.d_        = 3
#
#    slsOuts2 = state_fdbk_sls(sys, slsParams)
#    [x2, u2] = simulate_system(sys, slsParams, slsOuts2, simParams)
#    plot_heat_map(x2, sys.B2*u2, 'Localized')
#
#    ## (3) approximate d-localized sls
#    slsParams.mode_     = SLSMode.ApproxDLocalized
#    slsParams.cSpeed_   = 1
#    slsParams.robCoeff_ = 10^3
#
#    slsOuts3 = state_fdbk_sls(sys, slsParams)
#    [x3, u3] = simulate_system(sys, slsParams, slsOuts3, simParams)
#    plot_heat_map(x3, sys.B2*u3, 'Approximately Localized')


if __name__ == '__main__':
    state_fdbk_example()