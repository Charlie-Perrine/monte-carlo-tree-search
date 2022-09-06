from cgi import test
import numpy as np
import random

import matplotlib.pyplot as plt

from mctspy.games.examples.tictactoe import TicTacToeGameState
from agent import RandomAgent, MCTSAgent

#TODO Understand the Tic3 env

#TODO Implement a play function



def play():
    """
    Playing a game of Tic Tac Toe with two agents
    """

    #Initializes a new game
    state = np.zeros((5, 5))
    win = None


    #Game Loop
    while win == None:
        #Initializes the board
        board_state = TicTacToeGameState(state, next_to_move=-1, win = win)
        print('board_state: ', '\n', board_state.board)

        mcts_agent = MCTSAgent()
        #MCTS plays the best move
        new_board_state = mcts_agent.move(board_state)

        #Checks victory
        win = new_board_state.game_result
        print('mcts_win: ', win)

        #New state
        state = new_board_state.board
        print('mcts_board: \n', state)


        #Random player
        enemy_board = TicTacToeGameState(state, next_to_move=1, win = win)

        #Picks a random valid move
        random_agent = RandomAgent()
        enemy_move = random_agent.move(enemy_board)

        #Plays the move
        new_enemy_board = enemy_board.move(enemy_move)

        #Checks victory status
        win = new_enemy_board.game_result
        print('enemy_win:', win)

        #New state
        state = new_enemy_board.board
        print('board_enemy: \n', state)




play()
