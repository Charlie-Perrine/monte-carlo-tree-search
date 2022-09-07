"""
A Parameters module in which we define all the
params we need in our model.
"""

class Parameters():

    def __init__(self) -> None:
        #Number of simulations performed to get the best action
        self.num_simulations = 10
        self.convolution_layers = 3
        self.buffer_size = 1000
        self.learning_rate = 0.001
        self.batch_size = 200


params = Parameters()
