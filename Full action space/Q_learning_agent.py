# Q-learning agent for full action space supports the following actions:
# 1) doubling-down
# 2) splitting 
# TODO: 
# 3) insurance 
# 4) surrender 

# do not forget to check allowed actions 

from agent import agent
import numpy as np
import random
import time 

class QAgent(agent):

    def __init__(self, alpha=0.01):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)

        self.Stand = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, Stand)
        self.Hit = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, Hit)
        self.Split = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, Split)
        self.Double = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, Doubling)

        # Q-values of terminal states
        self.Stand[0], self.Stand[1], self.Stand[2] = 1, -1, 0 
        self.Hit[0], self.Hit[1], self.Hit[2] = 1, -1, 0 
        self.Split[0], self.Split[1], self.Split[2] = 1, -1, 0 
        self.Double[0], self.Double[1], self.Double[2] = 1, -1, 0 
        
        self.gamma = 1 # we have a single reward at the end => no need to discount anything 
        self.alpha = alpha # learning rate

    def policy(self, hand, allowed_actions): # given hand and allowed actions, take a certain action 
        state_index = self.state_approx(hand) # state index of the hand
        Q_stand = self.Stand[state_index]
        Q_hit = self.Hit[state_index]
        Q_split = self.Split[state_index]
        Q_double = self.Double[state_index]
       
        # Q-values of all actions for the given state 
        Q_values = {'stand': [Q_stand], 'hit': [Q_hit], 'split': [Q_split], 'double': [Q_double]}  
       
        # Q-values of allowed actions 
        Q_values_allowed = {}
        for key in Q_values:
            if key in allowed_actions:
                Q_values_allowed[key] = Q_values[key]

        # Q-values with biggest values
        Q_values_max = {}
        while True:
            key_max = max(Q_values_allowed, key=Q_values_allowed.get) # the action with biggest Q-value   
            Q_values_max[key_max] = Q_values_allowed.pop(key_max) 
            if Q_values_allowed:
                if Q_values_max[key_max] > max(Q_values_allowed.values()): # if Q values of other allowed actions are less
                    break 
            else:
                break

        action = random.choice(list(Q_values_max)) # we choose random action among those with highest Q values
        return action 


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
            next_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index 
            action = actions.pop(0) # the action which was done in the current state 

            Q_max = max(self.Hit[next_state_index], self.Stand[next_state_index], self.Split[next_state_index], 
                    self.Double[next_state_index]) # the max Q value for the next state 
            
            # and just perform updates of the corresponding Q function
            if action == 'hit': 
                self.Hit[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Hit[current_state_index])
            
            elif action == 'stand': 
                self.Stand[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Stand[current_state_index])

            elif action == 'split':
                self.Split[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Split[current_state_index])

            elif action == 'double': 
                self.Double[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Double[current_state_index])


    def get_Q(self):
        return self.Stand, self.Hit, self.Split, self.Double

if __name__ == '__main__':
    agent = QAgent() 
