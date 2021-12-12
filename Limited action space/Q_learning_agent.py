# Things that can be done here: 
# 1) it would be cool to visualize the policy (as .csv file) and include this table in the report
# 2) try alpha decay in order to converge to optimal (table) policy 
# 3) try online version of Q_learning 
# 4) try to see what would change if we set state-action value functions for terminal states to be different from [1, -1, 0];
#    actually in true Q-learning algorithm Q values of terminal states should be always 0!
# 5) try e-greedy policy instead of greedy policy for better exploration (crucial if we want to converge to the table policy!)
# 6) is it good to learn nothing from episodes when we have blackjack? 

from agent import agent
import numpy as np
import random

class QAgent(agent):

    def __init__(self, alpha):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.epsilon = 0.2 # GLIE policy parameter
        self.S = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, S)
        self.H = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, H)
        self.H[0], self.H[1], self.H[2] = 1, -1, 0 # Q(State,H) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.S[0], self.S[1], self.S[2] = 1, -1, 0 # Q(State,S) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.gamma = 1 # we have a single reward at the end => no need to discount anything 
        self.alpha = alpha # learning rate; TODO: 1) perform hyperparameter optimization; 2) see what happens if alpha decays in time ->
        # -> to guarantee convergence of the state-action value function; now it doesn't converge to 43.5% win rate of the optimal (table) policy  

    def policy(self, hand):
        # Q-learning is an off-policy TD control method and can work in both online/offline regimes.
        # Here, we implement offline regime, meaning that the policy updates only at the end of an episode. 
        # What would happen in the online regime? 

       # if np.random.rand() < self.epsilon:
        #    return random.choice(['h', 's'])
        state_index = self.state_approx(hand) # we return the current state index
        if self.H[state_index] > self.S[state_index]: # if Q(State, H) > Q(State, S) then we hit 
            return 'h'
        elif self.H[state_index] < self.S[state_index]:
            return 's'
        elif self.H[state_index] == self.S[state_index]:
            return random.choice(['h', 's'])

    def learn(self, episode):
        
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions: # actions list can be empty if either agent or dealer has a blackjack => nothing to learn here, just return 
            return

        if len(actions) != len(agent_hands): # then and only then the agent busted 
            del agent_hands[-1] # so we remove the last hand of the agent from the episode because it doesn't carry any useful information (it is just lose state)

        if reward > 0: # if the reward was positive, we won
            final_state_index = 0 # the index of the terminal win state
        elif reward < 0:
            final_state_index = 1 # the index of the terminal lose state
        elif reward == 0:
            final_state_index = 2 # the index of the terminal draw state  
       
        while agent_hands: # while there is something we can learn from 

            current_agent_hand = agent_hands.pop(0) # current hand 
            next_agent_hand = agent_hands[0] if agent_hands else None # next hand
            
            current_state_index = self.state_approx([current_agent_hand, dealer_card]) # index of the current state 
            next_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index # next state can be either the state 
            # correposponding to the next_agent_hand or it can be the terminal_state 

            action = actions.pop(0) # the action which was done in the current state 
            
            if action == 'h': # if the action was hit the we update corresponding Q(State, H) function
                self.H[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.H[
                    current_state_index])
            
            elif action == 's': # if the action was state the we update corresponding Q(State, S) function
                self.S[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[
                    current_state_index])

            # uncomment if we want alpha decay self.alpha = (1/ (1/alpha + 1))
            # example if the current alpha is 1/2 then the new alpha becomes 1/3
            #self.alpha = self.alpha/(1 + self.alpha)

    def get_Q(self):
        return self.H, self.S
