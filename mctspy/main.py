import numpy as np
import random

from agent import RandomAgent, MCTSAgent
from game import TicTacToeGameState

mcts = MCTSAgent()
rand = RandomAgent()

def play():
    """
    Playing a game of Tic Tac Toe with two agents
    """

    env = TicTacToeGameState(np.zeros((3, 3)))

    while True:

        action = mcts.move(env)
        env = env.move(action)

        if env.is_game_over():
            break

        action = rand.move(env)
        env = env.move(action)

        if env.is_game_over():
            break

    mcts.reset()

for _ in range(100):
    play()