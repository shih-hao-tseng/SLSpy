from slspy import *

def robust_simulations():
    # specify system matrices
    sys = LTISystem (
        Nx = 50, Nw = 50
    )

    # generate sys._A, sys._B2
    generate_doubly_stochastic_chain (
        system_model = sys,
        rho = 1,
        actuator_density = 0.5,
        alpha = 0.4
    )
    generate_BCD_and_zero_initialization(sys)

    synthesizer = SLS (
        system_model = sys,
        FIR_horizon = 10
    )
    # add objective
    synthesizer += SLSObj_H2 ()
    # add constraints
    approx_dlocalized = SLSCons_ApproxdLocalized (
        actDelay = 1,
        d = 6,
        robCoeff = 10e3
    )
    synthesizer += approx_dlocalized

    sim_horizon = 25
    simulator = Simulator (
        system = sys,
        horizon = sim_horizon
    )
    # generate noise
    noise = FixedNoiseVector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nw/2] = 10
    sys.useNoiseModel (noise_model = noise)

    cSpeeds = [2, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    cPrints = [2, 1, 0.4]  # which comm speeds to simulate & plot

    clnorms     = []
    robustStabs = []
    for cSpeed in cSpeeds:
        approx_dlocalized._cSpeed = cSpeed
        controller = synthesizer.synthesizeControllerModel()

        clnorms.append(synthesizer.getOptimalObjectiveValue())
        robustStabs.append(approx_dlocalized.getStabilityMargin())

        if cSpeed in cPrints:
            # initialize
            noise.startAtTime(0)

            # run the simulation
            simulator.setController (controller=controller)
            x_history, y_history, z_history, u_history = simulator.run ()

            Bu_history = matrix_list_multiplication(sys._B2,u_history)
            plot_heat_map(x_history, Bu_history, 'Comms = %d' % cSpeed)

    plot_line_chart(
        list_x=cSpeeds,
        list_y=clnorms,
        title='%d Node Chain' % sys._Nx,
        xlabel='Comm Speed',
        ylabel='Localized H_2-Norm Cost'
    )

    plot_line_chart(
        list_x=cSpeeds,
        list_y=robustStabs,
        title='%d Node Chain' % sys._Nx,
        xlabel='Comm Speed',
        ylabel='Stability Margin'
    )

if __name__ == '__main__':
    robust_simulations()
    keep_showing_figures()