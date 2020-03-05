from slspy import *

def state_fdbk_example():
    sys = LTI_System (
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
    # generate noise
    noise = Fixed_Noise_Vector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = Zero_Noise)
    noise._w[0][sys._Nw//2] = 10

    simulator = Simulator (
        system = sys,
        noise = noise,
        horizon = sim_horizon
    )

    ## (1) basic sls (centralized controller)
    # use SLS controller synthesis algorithm
    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 20
    )
    # set SLS objective
    synthesizer << SLS_Obj_H2()

    # synthesize controller (the generated controller is actually initialized)
    # and use the synthesized controller in simulation
    simulator.setController (
        controller = synthesizer.synthesizeControllerModel ()
    )

    # run the simulation
    x_history, y_history, z_history, u_history, w_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Centralized')


    ## (2) d-localized sls
    dlocalized = SLS_Cons_dLocalized (
        act_delay = 1,
        comm_speed = 2,
        d = 3
    )
    synthesizer << dlocalized

    simulator.setController (
        controller = synthesizer.synthesizeControllerModel ()
    )

    x_history, y_history, z_history, u_history, w_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Localized')


    ## (3) approximate d-localized sls
    approx_dlocalized = SLS_Cons_ApproxdLocalized (
        base = dlocalized,
        rob_coeff = 10e3
    )
    approx_dlocalized._cSpeed = 1

    # set the constriant
    synthesizer << approx_dlocalized

    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (controller=controller)

    x_history, y_history, z_history, u_history, w_history = simulator.run ()

    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    plot_heat_map(x_history, Bu_history, 'Approximately Localized')


if __name__ == '__main__':
    state_fdbk_example()
    keep_showing_figures()