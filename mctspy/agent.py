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
        self.mcts = MonteCarloTreeSearch(None)
        self.last = None, None
        self.nodes = {}

    def reset(self):
        self.last = None, None

    def move(self, env):
        """
        Returns the best state (T3 format)
        """
        if (key := repr(env)) not in self.nodes:
            parent, action = self.last
            node = Node(env, action, parent)
            self.nodes[key] = node
        else:
            node = self.nodes[key]

        self.mcts.root = node
        best_node = self.mcts.best_action(params.num_simulations)
        action = best_node.action

        # TODO Reset this on game_over
        self.last = node, action

        return best_node.action
