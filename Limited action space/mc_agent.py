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

class mc_agent(agent):
    def __init__(self):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.Q_S = np.zeros(self.NUMBER_OF_STATES)
        self.Q_H = np.zeros(self.NUMBER_OF_STATES)
        self.Q_S_count = np.zeros(self.NUMBER_OF_STATES) # count state-action visitations 
        self.Q_H_count = np.zeros(self.NUMBER_OF_STATES) # count state-action visitations 

    def policy(self, hand): # greedy policy based on the current estimate of Q functions
        state_index = self.state_approx(hand) 
        if self.Q_H[state_index] > self.Q_S[state_index]:
            return 'h'
        elif self.Q_H[state_index] < self.Q_S[state_index]:
            return 's'
        elif self.Q_H[state_index] == self.Q_S[state_index]:
            return random.choice(['h', 's'])

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

