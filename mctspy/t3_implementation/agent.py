"""
Agents
"""

import random

from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
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

    def move(self, board):
        """Returns a random move (np array)."""
        return random.choice(board.get_legal_actions())

class MCTSAgent(Agent):
    """
    MCTS Agent.
    """

    def __init__(self) -> None:
        super().__init__()

    def move(self, board):
        """Returns the best state (T3 format)"""
        #Current root
        current_root = TwoPlayersGameMonteCarloTreeSearchNode(state = board)

        #MCTS TODO Are we resetting the MCTS
        mcts = MonteCarloTreeSearch(current_root)

        #Picks best node
        best_node = mcts.best_action(params.num_simulations)

        return best_node.state
