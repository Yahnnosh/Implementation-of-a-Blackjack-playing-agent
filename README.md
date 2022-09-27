# Implementation of a Blackjack playing agent

# Introduction
### Motivation
The goal of this project is the development of an agent capable of playing the game of Blackjack
with a positive net return (i.e. "beat the house") using reinforcement learning. For this purpose,
different policies are tested in two different action spaces. Furthermore, we present a novel
dynamic betting algorithm that enables the policy to additionally adapt the betting amount in 
each round. For more information see our [project report](Project%20report.pdf).

# Methods
We use two different action spaces for our static betting (i.e. betting the same amount
each round) policies:
1. **Limited action space**: Only two of the allowed actions in Blackjack are used: _hit_ and _stand_
2. **Full action space**: All allowed actions of Blackjack are used: _hit, stand, double, split_ and _insurance_

### Limited action space
In this action space the following (static betting) policies have been tested:
1. Value iteration
2. Monte Carlo learning
3. Q-learning
4. Double Q-learning
5. SARSA learning
6. Deep Q-network

### Full action space
In this action space the following (static betting) policies have been tested:
1. Q-learning
2. SARSA learning

### Exploration policies
During training of the aforementioned algorithms, the following exploration policies have been
used to trade-off exploration and exploitation:
1. Random policy
2. Greedy policy
3. Epsilon-greedy policy
4. Upper confidence bound (UCB) policy
5. Boltzmann policy

## Dynamic betting
In this setting the static betting strategy (one of the policies from above) from one of the 
action spaces is augmented with our RL dynamic betting policy (see our [project report](Project%20report.pdf)
for more detail), using (among other information) card counting.

# Results
For smaller decks (which are advantageous when using card counting), our method is able to provide
large positive net return, i.e. "beats the house" and also surpasses conventional methods used by
professional Blackjack players. For larger decks, our method has not been able to provide 
positive net return yet. We believe the cause of this lies in the (relatively) poor convergence of 
the static betting policies when trained in the full action space.