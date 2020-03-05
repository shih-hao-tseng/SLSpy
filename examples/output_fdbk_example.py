from slspy import *

def output_fdbk_example():
    sys = LTI_System (
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
    # generate noise
    noise = FixedNoiseVector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nw//2] = 10

    simulator = Simulator (
        system = sys,
        noise = noise,
        horizon = sim_horizon
    )

    # use SLS controller synthesis algorithm
    # notice that the system should also be output-feedback (state_feedback = False)
    # to actually trigger the output-feedback SLS
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 20,
        state_feedback = False # use output-feedback synthesizer
    )

    # assume uncorrelated measurement noise
    mmNoiseAmp = 1.0
    cov_vSqrt  = mmNoiseAmp * np.eye(10)
    
    # set SLS objective    
    synthesizer << SLS_Obj_LQ(Cov_vSqrt=cov_vSqrt)
   
    # synthesize controller (the generated controller is actually initialized)
    # and use the synthesized controller in simulation
    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (
        controller = controller
    )

    # run the simulation
    x_history, _, _, u_history, _ = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Centralized')

if __name__ == '__main__':
    output_fdbk_example()
    keep_showing_figures()