from slspy import *
import numpy as np

def synthesis_algorithm_example():
    # specify system matrices
    sys = LTI_FIR_System (
        Ny = 5, Nu = 5
    )

    sys._G.append(np.eye(5))
    sys._G.append(np.eye(5))

    synthesizer = IOP (
        system_model = sys,
        FIR_horizon = 15
    )
    synthesizer << IOPObj_H2()

    # synthesize controller (the generated controller is actually initialized)
    # and use the synthesized controller in simulation
    simulator = Simulator (
        system = sys,
        horizon = 25
    )
    simulator.setController (
        controller = synthesizer.synthesizeControllerModel ()
    )

    # run the simulation
    _, y_history, _, u_history, _ = simulator.run ()

    plot_heat_map(y_history, u_history, 'IOP')

if __name__ == '__main__':
    synthesis_algorithm_example()
    keep_showing_figures()