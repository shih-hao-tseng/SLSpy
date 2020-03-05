from slspy import *
import numpy as np

def synthesis_algorithm_example():
    # specify system matrices
    sys = LTI_FIR_System (
        Ny = 10, Nu = 10
    )

    sys._G.append(np.eye(sys._Ny))
    sys._G.append(np.eye(sys._Ny))

    synthesizer = IOP (
        system_model = sys,
        FIR_horizon = 10
    )
    synthesizer << IOPObj_H2()

    sim_horizon = 25
    noise = FixedNoiseVector (Nw = sys._Ny, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nw//2] = 10

    # synthesize controller (the generated controller is actually initialized)
    controller = synthesizer.synthesizeControllerModel ()

    # and use the synthesized controller in simulation
    simulator = Simulator (
        system = sys,
        controller = controller,
        noise = noise,
        horizon = sim_horizon
    )

    # run the simulation
    _, y_history, _, u_history, _ = simulator.run ()

    plot_heat_map(y_history, u_history, 'IOP')

if __name__ == '__main__':
    synthesis_algorithm_example()
    keep_showing_figures()