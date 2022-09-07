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
        print(type(action))
        agent1_train = (env.board, action, env.game_result)

        if env.is_game_over():
            break

        action = mcts.move(env)
        env = env.move(action)

        agent2_train = (env.board, action, env.game_result)

        if env.is_game_over():
            break
    mcts.reset()
    #result = env.game_result
    #return result
    return agent1_train, agent2_train

#res = []
for _ in range(1):
    r1, r2 = play()
    #res.append(r)
    #print('Game Over \n\n')
#print(f'wins{res.count(1)} losses{res.count(-1)} draws{res.count(0)}')
print(r1)
print(r2)

score_1 = (234, 233, 533)
score_2 = (216, 191, 593)
score_3 = (215, 226, 559)
score_4 = (225, 240, 535)
score_5 = (106, 21, 9873)

def train():

    # 1. Init a buffer that contains 10000 positions (board, action, value) & INIT MCTS + NN agents
    BUFFER = collections.deque(maxlen=10000)
    NN = A2CAgent()
    MCTS = MCTSAgent(NN) # TODO init MCTS agent with a neural network

    # DONE 2. Call play to generate positions using self.play with MCTS (btw. for chess, take max 30 positions by game to avoid overfitting)
    for _ in range(1000):
        positions1, positions2 = play()
        BUFFER.append(positions1)
        BUFFER.append(positions2)

    # 3. Train the network using the buffer:
    for _ in range(200):
        A2CAgent().train(BUFFER) # TODO Code an agent to handle the training process with NN

    # 4. After 200 training steps, update MCTS with the new NN
    MCTS.update(NN) # TODO update MCTS with the new NN

    # 5. Start over.
    return BUFFER
