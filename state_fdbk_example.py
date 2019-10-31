from sls_sim.SystemModel import LTISystem
from sls_sim.Simulator import Simulator
from sls_sim.SynthesisAlgorithm import SLS
from sls_sim.NoiseModel import *
from sls_sim.PlantGenerator import *
import numpy as np

def state_fdbk_example():
    sys = LTISystem(
        Nx=10, Nw=10
    )

    # generate sys._A, sys._B2
    GenerateDoubleStochasticChain(
        system_model = sys,
        rho = 1,
        actuator_density = 1,
        alpha = 0.2
    )

    # specify system matrices
    sys._B1  = np.eye(sys._Nx)
    sys._C1  = np.stack( (np.eye(sys._Nx),np.zeros([sys._Nu,sys._Nx])), axis=0)
    sys._D12 = np.stack( (np.zeros([sys._Nx,sys._Nu]),np.eye(sys._Nu)), axis=0)

    sim_horizon = 25
    simulator = Simulator (
        system = sys,
        horizon = sim_horizon
    )

    noise = FixedNoiseVector(Nw=sys._Nx,horizon=sim_horizon)
    noise.generateNoiseFromNoiseModel(cls=ZeroNoise)
    noise._w[0][sys._Nx/2] = 10

    sys.useNoiseModel(noise_model=noise)

    ## (1) basic sls (centralized controller)
    synthesizer = SLS(
        FIR_horizon=20
    )

    synthesizer.setSystemModel(sys)

    controller = synthesizer.synthesizeControllerModel()

    simulator.setController (controller=controller)

    sys.initialize(x0=np.zeros([sys._Nx,1]))
    controller.initialize()
    x,y,z,u = simulator.run ()


#    
#    # sls parameters
#    slsParams       = SLSParams
#    slsParams.obj_  = Objective.H2 # objective function
#
#    # simulation parameters
#    simParams           = SimParams
#    simParams.openLoop_ = false
#
#    ## (1) basic sls (centralized controller)
#    slsParams.mode_ = SLSMode.Basic
#
#    slsOuts1 = state_fdbk_sls(sys, slsParams)
#    [x1, u1] = simulate_system(sys, slsParams, slsOuts1, simParams)
#    plot_heat_map(x1, sys.B2*u1, 'Centralized')
#
#    ## (2) d-localized sls
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