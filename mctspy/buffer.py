from collections import deque
import random

import numpy as np
import torch

from parameters import params

class Buffer():
    """
    Takes our data (board, action, value) and transforms it into the
    right format for our A2C Network.
    Board: np.array -> torch
    Action: T3 format -> tuple (move location, 1 or -1)
    Value: Replaces the Nones with 0s
    """

    def __init__(self) -> None:
        self.buffer = deque(maxlen = params.buffer_size)

    def processing_single(self, train_example):
        """
        Takes a single train_example (board, action, value) and transforms it into the
        right format for our A2C Network. Then stores it into our buffer
        Board: np.array -> torch
        Action: T3 format -> tuple (move location, 1 or -1)
        Value: Replaces the Nones with 0s
        """

        board, action, value = train_example

        #Returns the board in format that pytorch/the A2C can read
        board = torch.permute(
            torch.unsqueeze(
            torch.tensor(board), dim = -1).float(),
            (2, 0, 1))

        #Transforms a x:1 y:1 v: -1 string into a tuple
        action = self.action_to_tuple(action)

        #game_results returns a None during the name, need a 0 for the loss
        if value == None:
            value = 0

        self.buffer.append((board, action, value))


    def action_to_tuple(self, action):
        """
        Takes an action in TicTacToeMove format (x, y, key)
        And returns a tuple with a position from 0 to 8 and the key that's
        being played by the current player.
        cf: Petting Zoo
        """
        #Takes the digits from the string and computes the position on the board
        digits_action = [int(x) for x in action if x.isdigit()]
        return (3 * digits_action[0] + digits_action[1], digits_action[2])


    def get(self):
        """
        Gets a batch of observations of size params.batch_size.
        """
        if len(self.buffer) < params.batch_size:
            raise Exception('Not enough data in the buffer')

        batch = random.sample(self.buffer, params.batch_size)
        board, action, value = zip(*batch)
        return board, action, value

BUF = Buffer()

#Tests
list_train = [(np.array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 1.]]), 'x:2 y:2 v:1', None), (np.array([[ 0.,  0.,  0.],
       [ 0.,  0., -1.],
       [ 0.,  0.,  1.]]), 'x:1 y:2 v:-1', None), (np.array([[ 0.,  0.,  0.],
       [ 0.,  0., -1.],
       [ 1.,  0.,  1.]]), 'x:2 y:0 v:1', None), (np.array([[ 0.,  0.,  0.],
       [ 0., -1., -1.],
       [ 1.,  0.,  1.]]), 'x:1 y:1 v:-1', None), (np.array([[ 0.,  0.,  0.],
       [ 0., -1., -1.],
       [ 1.,  1.,  1.]]), 'x:2 y:1 v:1', 1)]

BUF.processing_single(list_train[1])
