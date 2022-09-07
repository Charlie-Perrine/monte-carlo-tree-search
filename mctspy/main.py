import numpy as np

from agent import MCTSAgent, A2CAgent
from buffer import BUF
from game import TicTacToeGameState
from parameters import params

mcts = MCTSAgent()

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

    # 1. Initialization
    A2C = A2CAgent()
    MCTS = MCTSAgent(A2C.policy)

    # 2. Call play to generate positions using self.play with MCTS
    # Append the buffer with the tuples (board, action, value)
    # (btw. for chess, take max 30 positions by game to avoid overfitting)
    for _ in range(params.training_episodes):
        trains = play()
        for i in range(len(trains)):
            BUF.processing_single(trains[i])

    # 3. Train the network using the buffer:
    # TODO Use BUF.get in the agent to train the NN.
    for _ in range(params.batch_size):
        A2C.train()

    # 4. After 200 training steps, update MCTS with the new NN
    # TODO update the MCTS with the new NN, write an update function
    MCTS.update(A2C.policy)

    # 5. Start over.
