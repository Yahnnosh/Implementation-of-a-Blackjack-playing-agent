# Following Sutton book "Reinforcement Learning" pages 93 - 95 and 99. They give a nice discussion of MC method specifically for the blackjack problem. 
# The algorithm I implement here is the "Monte Carlo ES" 
# the win rate is around 40.5% (vs 43.5% optimal)

# 1) arbitrary initialization of Q(s,a) and Returns(s,a)
# Loop: 
# 2) in the episode we act according to the policy <- argmax Q(s,a)
# 3) for each (S, A) appearing in an episode, we append the total reward of the episode to Returns(S, A)
# 4) Then we update Q function for each (S, A) pair from 3) as average of Returns(S,A)

# This method works good if all (S,A) pairs are visited often
# In our case I suspect it is not true. 
# This method is more suitable for simulations when we can generate episode starting from (S,A) and we pick this pair at random 
# So we can implement this later 
# Another idea is again (as in SARSA/Q_learning) to encourage the exploration by using some some sort of GLIE policy 

# Things to do: 
# 1) simulations
# 2) GLIE policy 

from agent import agent
import numpy as np
import random
import statistics
import math

class mc_agent(agent):
    def __init__(self, strategy='greedy', epsilon=0.5,
                 epsilon_decay=0.99999, temperature=5, ucb_param=2**0.5):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.Q_S = np.zeros(self.NUMBER_OF_STATES)
        self.Q_H = np.zeros(self.NUMBER_OF_STATES)
        self.Q_S_count = np.zeros(self.NUMBER_OF_STATES) # count state-action visitations 
        self.Q_H_count = np.zeros(self.NUMBER_OF_STATES) # count state-action visitations

        # Policy params
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') \
               or (strategy == 'ucb')
        self.strategy = strategy  # policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.ucb_param = ucb_param

    def policy(self, hand):
        # state approximation
        state_index = self.state_approx(hand)

        # random policy
        if self.strategy == 'random':
            return random.choice(['h', 's'])

        # greedy policy
        elif self.strategy == 'greedy':
            if self.Q_H[state_index] > self.Q_S[state_index]:  # if Q(State, H) > Q(State, S) then we hit
                return 'h'
            elif self.Q_H[state_index] < self.Q_S[state_index]:
                return 's'
            elif self.Q_H[state_index] == self.Q_S[state_index]:
                return random.choice(['h', 's'])

        # softmax policy
        elif self.strategy == 'softmax':
            def softmax(x):
                temperature = 1
                denom = sum([math.exp(temperature * q) for q in x])
                return [math.exp(temperature * q) / denom for q in x]

            action = random.choices(population=['h', 's'],
                                    weights=softmax([self.Q_H[state_index], self.Q_S[state_index]]))
            return action[0]

        # epsilon-greedy policy
        elif self.strategy == 'e-greedy':
            # act randomly
            if np.random.rand() < self.epsilon:
                action = random.choice(['h', 's'])
            # act greedily
            else:
                if self.Q_H[state_index] > self.Q_S[state_index]:  # if Q(State, H) > Q(State, S) then we hit
                    action = 'h'
                elif self.Q_H[state_index] < self.Q_S[state_index]:
                    action = 's'
                elif self.Q_H[state_index] == self.Q_S[state_index]:
                    action = random.choice(['h', 's'])

            self.epsilon *= self.epsilon_decay
            return action

        # UCB
        elif self.strategy == 'ucb':
            state_visitations = max(1, self.Q_H_count[state_index] + self.Q_S_count[state_index])
            hit_visitations = max(1, self.Q_H_count[state_index])
            stand_visitations = max(1, self.Q_S_count[state_index])

            # UCB
            Q_hit = self.Q_H[state_index] + self.ucb_param * (np.log(state_visitations) / hit_visitations) ** 0.5
            Q_stand = self.Q_S[state_index] + self.ucb_param * (np.log(state_visitations) / stand_visitations) ** 0.5

            if Q_hit == Q_stand:
                return random.choice(['h', 's'])
            else:
                return 'h' if Q_hit > Q_stand else 's'

        else:
            raise NotImplementedError

    def activate(self, strategy):
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') \
               or (strategy == 'ucb') or (strategy == 'table')
        self.strategy = strategy

    def learn(self, episode):
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions: # actions list can be empty if either agent or dealer has a blackjack => nothing to learn here, just return 
            return

        if len(actions) != len(agent_hands): # then and only then the agent busted 
            del agent_hands[-1] # so we remove the last hand of the agent from the episode because it doesn't carry any useful information for learning 

        if reward > 0: # if the reward was positive, we won
            final_state_index = 0 # the index of the terminal win state
        elif reward < 0:
            final_state_index = 1 # the index of the terminal lose state
        elif reward == 0:
            final_state_index = 2 # the index of the terminal draw state  

        while agent_hands: # loop over all hands in the episode 
            agent_hand = agent_hands.pop(0) # each hand
            action = actions.pop(0) # corresponding action
            state_index = self.state_approx([agent_hand, dealer_card]) # index of each state encountered in the episode
            if action == 'h':
                self.Q_H[state_index] = (self.Q_H[state_index] * self.Q_H_count[state_index] + reward) / (self.Q_H_count[state_index] + 1) # mean update 
                self.Q_H_count[state_index] += 1 # count how many time we updated this state-action q function
            elif action == 's':
                self.Q_S[state_index] = (self.Q_S[state_index] * self.Q_S_count[state_index] + reward) / (self.Q_S_count[state_index] + 1) 
                self.Q_S_count[state_index] += 1 

    def get_Q(self):
        return self.Q_H, self.Q_S

    def get_Q_hand(self, hand):
        state_index = self.state_approx(hand)
        return self.Q_H[state_index], self.Q_S[state_index]

    def get_visitations(self, hand):
        state_index = self.state_approx(hand)
        return self.Q_H_count[state_index], self.Q_S_count[state_index]
