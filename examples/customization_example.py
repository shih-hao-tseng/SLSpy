from slspy import *
# import the bases of controller model and synthesis algorithm
from slspy.core import ControllerModel, SynthesisAlgorithm
import numpy as np

# define a new controller model
class MyControllerModel (ControllerModel):
    def __init__(self, Nu):
        # this model takes a variable Nu
        self._Nu = Nu

    # a controller model must have a method called
    #    getControl(self,y)
    # which takes the system measurement y and generates the control u
    def getControl(self,y):
        # and generate zero vector of dimension Nu as the output
        u = np.zeros([self._Nu,1])
        # return the control
        return u

# define a new synthesis algorithm
class MySynthesisAlgorithm (SynthesisAlgorithm):
    # a synthesis algorithm must have a method called
    #    synthesizeControllerModel(self)
    # which uses its parameters and generate a controller model
    def synthesizeControllerModel(self):
        # In this case, we return an instance of MyControllerModel
        controller = MyControllerModel(Nu=10)
        return controller

def customization_example():
    '''
    This is a simple example showing how to customize a controller model
    and a synthesis algorithm, the simple controller model returns zero
    vector as the control.
    '''
    # we use a simple LTI system with doubly stochastic chain topology
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
    # some other matrix assignments
    generate_BCD_and_zero_initialization(sys)

    sim_horizon = 25
    # generate noise for the system model sys
    # which is an impulse at time 0
    # NoiseModel is, again, customizable, and the base class is NoiseModel
    # here we use a predefined noise model FixedNoiseVector
    noise = FixedNoiseVector (Nw = sys._Nw, horizon = sim_horizon)
    noise.generateNoiseFromNoiseModel (cls = ZeroNoise)
    noise._w[0][sys._Nw//2] = 10

    # we need a simulator to run the simulation
    # a Simulator takes a SystemModel (system), a ControllerModel (controller) and a simulation horizon (horizon)
    # to run the simulation
    simulator = Simulator (
        system = sys,
        noise = noise,
        horizon = sim_horizon
    )

    # create a synthesizer
    synthesizer = MySynthesisAlgorithm (
        system_model = sys
    )

    # synthesize the controller model and assign the controller model to the simulator
    controller = synthesizer.synthesizeControllerModel ()
    simulator.setController (
        controller = controller
    )

    # run the simulation and get the histories of
    #   x_history: the state
    #   y_history: the measurement
    #   z_history: the output
    #   u_history: the control
    #   w_history: the noise
    x_history, _, _, u_history,_ = simulator.run ()

    # we are interested in the quantity Bu
    Bu_history = matrix_list_multiplication(sys._B2,u_history)
    # plot x v.s. Bu
    plot_heat_map(x_history, Bu_history, 'Centralized')

if __name__ == '__main__':
    customization_example()
    keep_showing_figures()