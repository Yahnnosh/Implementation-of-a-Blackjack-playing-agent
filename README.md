# Implementation of a Blackjack playing agent
The goal of this project is the development of an agent capable of playing the game of Blackjack with a positive net return (i.e. "beat the house") using reinforcement learning.
## Limited action space
In this setting the action space is limited to the binary action space A'={hit, stand}.
Contains:
- **agent.py**: interface for all agents
- **human_player.py**: human input from command line
- **dealer_policy.py**: agent that follows the same policy as the dealer, i.e. hit for hands lower than 17, else stand. Hits soft 17.
- **dealer.py**: Blackjack environment
- **random_policy.py**: agent that acts randomly
- **table_policy.py**: agent that acts according to the fixed policy from soft_table.csv and hard_table.csv
- **card_counting.py**: agent that acts according to the fixed policy from soft_table.csv and hard_table.csv and also counts cards using Hi_Lo strategy
- **evaluate_performance.py**: evaluation of policy performance based on emp. mean win rate and emp. long term profitability
- **model_based_agent.py**: interface for model-based agents
- **value_iteration.py**: agent that performs VI on each round
- **fast_value_iteration.py**: same agent as value_iteration.py but faster (computes only reachable state space)
- **Q_learning_agent.py**: agent that performs offline Q-learning (see page 131 in Sutton book) 
- **sarsa_agent.py**: agent that performs online SARSA control (see page 130 in Sutton book) 
- **mc_agent.py**: agent that performs Monte Carlo ES (see page 99 in Sutton book)
- **get_table_from_agent.py**: plots the tables for all card pairs (hard, soft) for a specific (model-based) policy

## Full action space
In this setting the action space is expanded to the full action space A={hit, stand, split, double, insurance}.
