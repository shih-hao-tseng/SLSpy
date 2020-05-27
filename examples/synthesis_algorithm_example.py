from slspy import *
import numpy as np
import cvxpy as cp

def synthesis_algorithm_example():
    # specify system matrices
    sys = LTI_System (
        Nx = 10, Nw = 10
    )

    # generate sys._A, sys._B2
    generate_doubly_stochastic_chain (
        system_model = sys,
        rho = 0.5,
        actuator_density = 1,
        alpha = 0.2
    )
    generate_BCD_and_zero_initialization(sys)

    sim_horizon = 25
    sys_FIR = truncate_LTI_System_to_LTI_FIR_System (system=sys, FIR_horizon=sim_horizon)

    controller_FIR_horizon = 10
    noise = FixedNoiseVector (Nw = sys._Nx, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nw//2] = 10

    # try SLS
    synthesizer_sls = SLS (
        system_model = sys,
        FIR_horizon = controller_FIR_horizon
    )
    synthesizer_sls << SLS_Obj_H2()
    controller_sls = synthesizer_sls.synthesizeControllerModel ()    
    
    simulator = Simulator (
        system = sys,
        controller = controller_sls,
        noise = noise,
        horizon = sim_horizon
    )
    _, y_history, _, u_history, _ = simulator.run ()
    
    plot_heat_map(y_history, u_history, 'SLS', left_title='log10(|y|)', right_title='log10(|u|)')

    # try different solver option
    # we can also write synthesizer_sls.getSolver().setOptions()
    CVX_solver = synthesizer_sls.getSolver()
    CVX_solver.setOptions(solver=cp.SCS) # try SCS
    controller_sls = synthesizer_sls.synthesizeControllerModel ()    
    
    simulator = Simulator (
        system = sys,
        controller = controller_sls,
        noise = noise,
        horizon = sim_horizon
    )
    _, y_history, _, u_history, _ = simulator.run ()
    
    plot_heat_map(y_history, u_history, 'SLS with SCS solver', left_title='log10(|y|)', right_title='log10(|u|)')

    CVX_solver.setOptions(solver=cp.ECOS) # try ECOS
    controller_sls = synthesizer_sls.synthesizeControllerModel ()    
    
    simulator = Simulator (
        system = sys,
        controller = controller_sls,
        noise = noise,
        horizon = sim_horizon
    )
    _, y_history, _, u_history, _ = simulator.run ()
    
    plot_heat_map(y_history, u_history, 'SLS with ECOS solver', left_title='log10(|y|)', right_title='log10(|u|)')

    # try IOP
    synthesizer_iop = IOP (
        system_model = sys_FIR,
        FIR_horizon = controller_FIR_horizon
    )
    synthesizer_iop << IOP_Obj_H2()
    controller_iop = synthesizer_iop.synthesizeControllerModel ()

    simulator.setSystem(sys_FIR)
    simulator.setController(controller_iop)
    _, y_history, _, u_history, _ = simulator.run ()

    plot_heat_map(y_history, u_history, 'IOP', left_title='log10(|y|)', right_title='log10(|u|)')

if __name__ == '__main__':
    synthesis_algorithm_example()
    keep_showing_figures()