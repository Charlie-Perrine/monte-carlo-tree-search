import collections
import numpy as np
import random

from agent import RandomAgent, MCTSAgent, A2CAgent
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
    NN = A2CAgent()
    MCTS = MCTSAgent(NN.policy) # TODO init MCTS agent with a neural network

    # 2. Call play to generate positions using self.play with MCTS (btw. for chess, take max 30 positions by game to avoid overfitting)
    # TODO Change the format of action into a list cf. action space pz
    # TODO change None values to 0
    for _ in range(1000):
        positions1, positions2 = play()
        NN.BUFFER.append(positions1)
        NN.BUFFER.append(positions2)

    # 3. Train the network using the buffer:
    # TODO Add the stuff to the buffer
    # TODO get the agent to take stuff from the buffer and feed it to the NN
    for _ in range(200):
        A2CAgent().train()

    # 4. After 200 training steps, update MCTS with the new NN
    MCTSAgent.update(NN.policy) # TODO update MCTS with the new NN

    # 5. Start over.
