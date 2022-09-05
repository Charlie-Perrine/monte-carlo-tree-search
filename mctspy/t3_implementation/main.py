from cgi import test
import numpy as np
import random

from mctspy.games.examples.tictactoe import TicTacToeGameState
from mctspy.tree.nodes import TwoPlayersGameMonteCarloTreeSearchNode
from mctspy.tree.search import MonteCarloTreeSearch
from parameters import params

#TODO Understand the Tic3 env

#TODO Implement a play function



def play():
    """
    Playing a game of Tic Tac Toe with two agents
    """

    #Initializes a new game
    state = np.zeros((3, 3))
    win = None

    #Game Loop
    while win == None: #TODO wont work cause i have two wins.... AH
        #Initializes the board
        board_state = TicTacToeGameState(state, next_to_move=1, win = win)
        print('board_state: ', board_state)

        #legal_moves = board_state.get_legal_actions()

        #Initiliazes the current root
        current_root = TwoPlayersGameMonteCarloTreeSearchNode(state = board_state)
        print('current_root: ', current_root)

        #TODO NN

        #MCTS
        mcts = MonteCarloTreeSearch(current_root)
        print('mcts: ', mcts)

        #Picks the best node
        best_node = mcts.best_action(params.num_simulations)
        print('best_node: ', best_node)

        #Plays the best node
        new_board_state = best_node.state
        print('new_board: ', new_board_state)

        #Checks if the game has ended
        win = new_board_state.game_result
        print('win_mcts: ', win)

        #New state
        state = new_board_state.board
        print('board', state)

        #Random agent
        enemy_board = TicTacToeGameState(state, next_to_move=1, win = win)
        enemy_move = random.choice(enemy_board.get_legal_actions())
        enemy_board = enemy_board.move(enemy_move)
        win = enemy_board.game_result
        print('win_enemy:', win)
        state = enemy_board.board
        print('board', state)




    #TODO how do we keep on playing?
    #Do we use petting zoo?
    #Whos the other agent?

    #TODO Implement an agent that picks a random move.


    #TODO Neural Network: how and when does it learn?

play()
