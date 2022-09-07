"""
Agents
"""

import collections
import random
import numpy as np

import torch

from tree import Node, MonteCarloTreeSearch
from parameters import params

from buffer import BUF
from network import A2CNet

class Agent:
    def __init__(self) -> None:
        pass

    def move(self, move):
        pass


class RandomAgent(Agent):
    """
    Returns a random move from the legal moves.
    """

    def __init__(self) -> None:
        super().__init__()

    def move(self, env):
        """
        Returns a random move (np array).
        """
        return random.choice(env.get_legal_actions())

class MCTSAgent(Agent):
    """
    MCTS Agent.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mcts = MonteCarloTreeSearch(None)
        self.last = None, None
        self.nodes = {}

    def reset(self):
        self.last = None, None

    def move(self, env):
        """
        Returns the best state (T3 format)
        """
        #If the current node (repr(env)) hasn't been visited yet
        #We initialize it with the parent node and action
        #Else we take the current node
        if (key := repr(env)) not in self.nodes:
            parent, action = self.last
            node = Node(env, action, parent)
            self.nodes[key] = node
        else:
            node = self.nodes[key]

        self.mcts.root = node
        best_node = self.mcts.best_action(params.num_simulations)
        action = best_node.action

        self.last = node, action

        return best_node.action

class A2CAgent(Agent):
    """
    A2C Agent
    """

    def __init__(self):
        super().__init__()

        self.idx = 0
        self.net = A2CNet()
        self.BUFFER = collections.deque(maxlen=params.buffer_size)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=params.learning_rate)
        self.policy = None


    def learn(self):
        """
        Trains the model.
        """
        #TODO

        #Getting the predictions and the targets
        board, action, value = BUF.get()
        val, pol = self.net(board)

        self.policy = pol

        #Policy and value predictions
        y_pred_pol = torch.log(pol)
        y_pred_val = val

        #Went for advs since we don't have the true policy
        y_true_val = value # TODO + params.gamma * self.net(new)
        adv = y_true_val - y_pred_val

        #Policy loss
        pol_loss = -(adv * y_pred_pol)
        #Val loss
        val_loss = 0.5 * torch.square(adv)
        #Total loss
        loss = (pol_loss + val_loss)#.mean()

        self.idx += 1

        #TODO ? is this the right thing to do?
        self.policy = pol

        #Backprop
        self.opt.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.net.pol.parameters(), 0.001)
        #torch.nn.utils.clip_grad_norm_(self.net.val.parameters(), 0.001)
        self.opt.step()

    def save(self, path: str):
        """
        Save the agent's model to disk.
        """
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        """
        Load the agent's weights from disk.
        """
        dat = torch.load(path, map_location=torch.device("cpu"))
        self.net.load_state_dict(dat)
