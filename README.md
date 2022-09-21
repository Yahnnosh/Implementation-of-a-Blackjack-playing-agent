# Implementation of a Blackjack playing agent
The goal of this project is the development of an agent capable of playing the game of Blackjack with a positive net return (i.e. "beat the house") using reinforcement learning.
## Limited action space
In this setting the action space is limited to the binary action space A'={hit, stand}.
Contains:

**Environment**
- **dealer.py**: Blackjack environment

**Baseline policies**
- **random_policy.py** (low baseline): agent with random action
- **dealer_policy.py** (medium baseline): agent that follows the same policy as the dealer, i.e. hit for hands lower than 17, else stand. Hits soft 17.
- **table_policy.py** (hard baseline): agent that acts according to the fixed policy from soft_table.csv and hard_table.csv
- **card_counting.py** (optimal baseline): agent that acts according to the fixed policy from soft_table.csv and hard_table.csv and also counts cards using Hi-Lo strategy

**Policies**
- **Monte Carlo learning**
- **SARSA learning**
- **Q-learning**
- **Double Q-learning**
- **DQN**
- **Value iteration**

**Additional functions**

## Full action space
In this setting the action space is expanded to the full action space A={hit, stand, split, double, insurance}.
Contains:

**Environment**
- **dealer.py**: Blackjack environment

**Baseline policies**
- **table_policy.py** (hard baseline): agent that acts according to the fixed policy from soft_table.csv, hard_table.csv and split_table.csv
- **card_counting.py** (optimal baseline): agent that acts according to the fixed policy from soft_table.csv, hard_table.csv and split_table.csv and also counts cards using Hi-Lo strategy

**Policies**
- **SARSA learning**
- **Q-learning**

**Additional functions**

## Dynamic betting
In this setting the static betting strategy from one of the action spaces is augmented to the full policy π = (π_static, π_dynamic).
Contains:

**Dynamic betting policies**
- **Hi-Lo**: policy following the Hi-Lo heuristic, commonly used by human players using card counting
- **Model-based**: Our novel RL dynamic betting algorithm, outperforming Hi-Lo

**Additional functions**