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
    Returns a list of tuples (board, action, game_result)
    """

    #Initializes a new game
    env = TicTacToeGameState(np.zeros((3, 3)))
    train_examples = []

    #Loops until the end of the game
    while True:

        #MCTS plays
        action = mcts.move(env)
        env = env.move(action)
        print(type(action))
        agent1_train = (env.board, str(action), env.game_result)
        train_examples.append(agent1_train)

        #Ends the loop if MCTS won
        if env.is_game_over():
            break

        #Second MCTS plays
        action = mcts.move(env)
        env = env.move(action)

        agent2_train = (env.board, str(action), env.game_result)
        train_examples.append(agent2_train)

        #Ends the loop if MCTS won
        if env.is_game_over():
            break

    #Resets the last node of the MCTS
    mcts.reset()
    return train_examples


for _ in range(1):
    examples = play()
print(examples)
print(type(examples[0][0]))

#Scores for 5x5 self-play
score_1 = (234, 233, 533)
score_2 = (216, 191, 593)
score_3 = (215, 226, 559)
score_4 = (225, 240, 535)
score_5 = (106, 21, 9873)


#Pseudo code / TODO s
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
