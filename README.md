## (IN PROGRESS) Monte Carlos Tree Search and Advantage Actor Critic


Python implementation of a [Monte Carlo Tree Search](https://int8.io/monte-carlo-tree-search-beginners-guide) (MCTS) and Advantage Actor Critic (A2C) to Tic Tac Toe.


### Overview

The goal of this project is to develop a Neural Network capable of playing Tic Tac Toe, with the ultimate goal of broadening it's scope to other, more complex games.
We decided to implement a reinforcement learning model since they work well when it comes to learning how to play games.


### Credits

The code for playing Tic Tac Toe and for the MCTS was adapted from [int8's](https://github.com/int8/monte-carlo-tree-search) implementation of the Monte Carlo Tree Search.


### Details of the Project

In the master branch, you can train a MCTS to play a game of Tic Tac Toe.
It can learn by playing against a random agent, or against it's own self. We found self-play to be much more effective than the random agent.
In the main.py file, the default setting is for a 3x3 Tic Tac Toe grid. But you can easily train it on a bigger grid by going into the play function and changing the environment's initialization from a np.zeros((3, 3)) to a bigger sized array.

The char branch is in progress. We are currently coding an A2C on top of the MCTS. The Monte Carlo learns really quickly in Tic Tac Toe, but we want to have a working A2C + MCTS on a simple game before moving onto more complicated games.


### Detailed Package Workflow

master branch:
- `main.py` is the entry point of the package. You can play Tic Tac Toe and train the MCTS.
- `agent.py` defines the different agents: the random agent that plays a random legal move and the MCTS Agent that runs a tree search in order to pick the best legal move.
- `game.py` was written by [int8](https://github.com/int8/monte-carlo-tree-search) and creates the Tic Tac Toe environment. We would need to adapt this file in order to play other games.
- `network.py` contains the A2C Neural network.
- `parameters.py` defines all of the parameters needed in the package.
- `tree.py` contains the main structure of the MCTS. It runs a tree search and picks the best node (action) given a board state.

char branch:
(In progress)


Feel free to reach out if you have any questions or comments!
