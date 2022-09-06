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

        print(env.board, '\n')

        if env.is_game_over():
            break

        action = mcts.move(env)
        env = env.move(action)

        print(env.board, '\n')

        if env.is_game_over():
            break
    result = env.game_result
    mcts.reset()
    return result

res = []
for _ in range(10000):
    r = play()
    res.append(r)
    print('Game Over \n\n')
print(f'wins{res.count(1)} losses{res.count(-1)} draws{res.count(0)}')
