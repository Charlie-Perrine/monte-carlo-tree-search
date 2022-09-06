import collections
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



def train():

    # 1. Init a buffer that contains 10000 positions (board, action, value) & INIT MCTS + NN agents
    BUFFER = collections.deque(maxlen=10000)
    NN = A2CAgent()
    MCTS = MCTSAgent(NN) # TODO init MCTS agent with a neural network

    # 2. Call play to generate positions using self.play with MCTS (btw. for chess, take max 30 positions by game to avoid overfitting)
    for _ in range(1000):
        positions = play() # TODO play() should return a list of positions (board, action, value)
        BUFFER.append(positions)

    # 3. Train the network using the buffer:
    for _ in range(200):
        A2CAgent().train(BUFFER) # TODO Code an agent to handle the training process with NN

    # 4. After 200 training steps, update MCTS with the new NN
    MCTS.update(NN) # TODO update MCTS with the new NN

    # 5. Start over.
