import numpy as np
from collections import defaultdict

import time

class MonteCarloTreeSearch(object):

    def __init__(self, node):
        """
        MonteCarloTreeSearchNode
        Parameters
        ----------
        node : Node
        """
        self.root = node

    def best_action(self, simulations_number=None, total_simulation_seconds=None):
        """

        Parameters
        ----------
        simulations_number : int
            number of simulations performed to get the best action

        total_simulation_seconds : float
            Amount of time the algorithm has to run. Specified in seconds

        Returns
        -------

        """

        if simulations_number is None :
            assert(total_simulation_seconds is not None)
            end_time = time.time() + total_simulation_seconds
            while True:
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                if time.time() > end_time:
                    break
        else :
            for _ in range(0, simulations_number):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
        # to select best child go for exploitation only
        return self.root.best_child(c_param=0.)

    def _tree_policy(self):
        """
        selects node to run rollout/playout for

        Returns
        -------

        """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

class Node():

    def __init__(self, state, action, parent=None, policy=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self._number_of_visits = 0.
        self._results = defaultdict(int) #TODO Figure out how the policy changes this??
        self._untried_actions = None
        #TODO change the policy bases on Actor Critic
        self.policy = 1 #policy - Left it as one for debugging.

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        # TODO Do we need to track this globally
        child_node = Node(next_state, action, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.q / c.n) + c_param * self.policy * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ] #TODO Check formula
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]


    def normalize(self, policy, mask):
        """
        Ensure a policy is a proper probability distribution.
        """
        # Apply the action mask to the policy
        policy = policy * mask

        # Make it equal to the mask if no actions are available
        if sum(policy) == 0:
            policy = mask.astype("float64")

        # Normalize the distribution
        policy /= sum(policy)
        return policy


    def getActionProb(self, env):
            """
            This function performs params.num_simulations simulations of MCTS
            starting from a state (board).

            Returns:
                a policy vector where the probability of the ith action is
                    proportional to number_of_visits
            Note: for chess, would need to add a tempcontrol after 30 moves.
            Since T3 isn't as complex, didn't add it here.
            """

            # Gets an action mask
            #Gets the coordinates for the empty spots in the grid
            indices_t3 = np.where(env.board == 0)
            #Stores the indexes in PZ format
            index = []
            for coords in list(zip(indices_t3[0], indices_t3[1])):
                index.append(3 * coords[0] + coords[1])
            #Create our mask
            mask = [0 for _ in range (9)]
            for i in index:
                mask[i] = 1

            exit()
            # TODO Run board search. Might need to create a train function.
            # Or could be a self.mcts.best_action . How to link the tree to the node?
            for i in range(self.args.numMCTSSims):
                self.search(canonicalBoard)

            #TODO turn it into the right format
            s = self.game.stringRepresentation(canonicalBoard)

            #TODO calculate the policy
            counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]

            #TODO
            counts = [x  for x in counts]
            #TODO rewrite the normalization function
            counts = self.normalize(counts, mask)
            return counts
