"""
Agents
"""

import random
import numpy as np

from tree import Node, MonteCarloTreeSearch
from parameters import params

class Agent:
    def __init__(self) -> None:
        pass

    def move(self, move):
        pass


class RandomAgent(Agent):
    """
    Returns a random move from the legal moves.
    """

    def __init__(self) -> None:
        super().__init__()

    def move(self, env):
        """
        Returns a random move (np array).
        """
        return random.choice(env.get_legal_actions())

class MCTSAgent(Agent):
    """
    MCTS Agent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mcts = None
        self.nodes = {}

    def move(self, env):
        """
        Returns the best state (T3 format)
        """
        if repr(env) not in self.nodes:
            node = Node()
        repr(env)
        # node = 
        exit()
        #Current root
        current_root = Node(state = board)

        #MCTS TODO Are we resetting the MCTS
        mcts = MonteCarloTreeSearch(current_root)

        #Picks best node
        best_node = mcts.best_action(params.num_simulations)

        return best_node.state
