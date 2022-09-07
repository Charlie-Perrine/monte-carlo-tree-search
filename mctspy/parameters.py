"""
A Parameters module in which we define all the
params we need in our model.
"""

class Parameters():

    def __init__(self) -> None:
        #Number of simulations performed to get the best action
        self.num_simulations = 10
        self.convolution_layers = 3


params = Parameters()
