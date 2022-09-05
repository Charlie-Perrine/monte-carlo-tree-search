"""
Agents
"""

import random

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
        return random.choice(board.get_legal_actions())

class MCTS(Agent):
    """
    MCTS Agent.
    """

    def __init__(self) -> None:
        super().__init__()

    def move(self, board):
        pass
