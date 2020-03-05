from slspy import *

def robust_simulations():
    # specify system matrices
    sys = LTI_System (
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
    # set objective
    obj_H2 = SLSObj_H2 ()
    synthesizer << obj_H2

    # add constraints
    # robustness constraint should be added before dlocalized as it modifies the SLS constriants
    robust = SLSCons_Robust (
        gamma_coefficient = 10e3
    )
    synthesizer << robust

    dlocalized = SLSCons_dLocalized (
        actDelay = 1,
        d = 6
    )
    synthesizer += dlocalized

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

    cSpeeds = [2, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    cPrints = [2, 1, 0.4]  # which comm speeds to simulate & plot

    clnorms     = []
    robustStabs = []
    for cSpeed in cSpeeds:
        dlocalized._cSpeed = cSpeed
        controller = synthesizer.synthesizeControllerModel()

        clnorms.append(obj_H2.getObjectiveValue())
        robustStabs.append(robust.getStabilityMargin())

        if cSpeed in cPrints:
            # run the simulation
            simulator.setController (controller=controller)
            x_history, _, _, u_history, _ = simulator.run ()

            Bu_history = matrix_list_multiplication(sys._B2,u_history)
            plot_heat_map(x_history, Bu_history, 'Comms = %d' % cSpeed)

    plot_line_chart(
        list_x=cSpeeds,
        list_y=clnorms,
        title='%d Node Chain' % sys._Nx,
        xlabel='Comm Speed',
        ylabel='Localized H_2-Norm Cost',
        invert_x=True
    )

    plot_line_chart(
        list_x=cSpeeds,
        list_y=robustStabs,
        title='%d Node Chain' % sys._Nx,
        xlabel='Comm Speed',
        ylabel='Stability Margin',
        invert_x=True
    )

if __name__ == '__main__':
    robust_simulations()
    keep_showing_figures()