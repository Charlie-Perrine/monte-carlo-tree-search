"""
Agents
"""

import collections
import random
import numpy as np

import torch

from tree import Node, MonteCarloTreeSearch
from parameters import params

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
        if (key := repr(env)) not in self.nodes:
            parent, action = self.last
            node = Node(env, action, parent)
            self.nodes[key] = node
        else:
            node = self.nodes[key]

        #print(f"{node}, childrens: {len(node.children)}\nVisits:{node._number_of_visits}, Value:{node._results}")

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


    def move(self, obs, _):
        """
        Next action selection.
        """
        # TODO
        mask = np.array([random.randint(0, 1) for _ in range(4672)])
        obs = torch.tensor(obs).float().unsqueeze(0)
        _, pol = self.net(obs)
        pol = pol.squeeze(0).detach().numpy() * mask
        pol = pol / sum(pol)
        return np.random.choice(range(len(pol)), p=pol)

    def learn(self):
        """
        Trains the model.
        """
        old, act, rwd, new = BUF.get()
        val, pol = self.net(old)

        entropy = (pol.detach() * torch.log(pol.detach())).sum(axis=1)

        y_pred_pol = torch.log(torch.gather(pol, 1, act).squeeze(1) + 1e-6)
        y_pred_val = val.squeeze(1)
        y_true_val = rwd + CFG.gamma * self.net(new)[0].squeeze(1).detach()
        adv = y_true_val - y_pred_val

        val_loss = 0.5 * torch.square(adv)
        pol_loss = -(adv * y_pred_pol)
        loss = (pol_loss + val_loss).mean()  # + 1e-6 * entropy

        self.idx += 1

        # print(y_pred_pol)
        tp = pol[0].detach()
        tps, _ = torch.sort(tp, descending=True)
        print(tp.max(), tp.mean(), tp.min())
        print(tps.numpy()[:5])
        #print(self.idx, pol_loss, loss)

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
