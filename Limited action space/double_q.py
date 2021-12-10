from agent import agent
import numpy as np
import random

class double_QAgent(agent):

    def __init__(self):
        self.NUMBER_OF_STATES = 363 
        
        self.Q1_S = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, S)
        self.Q1_H = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, H)

        self.Q2_S = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, S)
        self.Q2_H = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, H)

        self.Q_S = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, S)
        self.Q_H = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, H)
        
        self.Q1_H[0], self.Q1_H[1], self.Q1_H[2] = 10, -10, 0 
        self.Q1_S[0], self.Q1_S[1], self.Q1_S[2] = 10, -10, 0 

        self.Q2_H[0], self.Q2_H[1], self.Q2_H[2] = 10, -10, 0 
        self.Q2_S[0], self.Q2_S[1], self.Q2_S[2] = 10, -10, 0 

        self.Q_H[0], self.Q_H[1], self.Q_H[2] = 0, 0, 0 
        self.Q_S[0], self.Q_S[1], self.Q_S[2] = 0, 0, 0 
        
        self.gamma = 1 
        self.alpha = 0.01 

        self.beta = 0.5 # probability (see double Q algo)

    def policy(self, hand):
        state_index = self.state_approx(hand) # we return the current state index
        if self.Q_H[state_index] > self.Q_S[state_index]: # if Q(State, H) > Q(State, S) then we hit 
            return 'h'
        elif self.Q_H[state_index] < self.Q_S[state_index]:
            return 's'
        elif self.Q_H[state_index] == self.Q_S[state_index]:
            return random.choice(['h', 's'])

    def update_q(self):
        self.Q_S = (self.Q1_S + self.Q2_S)/2
        self.Q_H = (self.Q1_H + self.Q2_H)/2

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


            if action == 'h': # if the action was hit the we update corresponding Q(State, H) function
                if np.random.rand() < self.beta: # then we update q1 
                    
                    if (self.Q1_S[next_state_index] > self.Q1_H[next_state_index]):
                        max_action = 's'
                    elif (self.Q1_S[next_state_index] < self.Q1_H[next_state_index]):
                        max_action = 'h'
                    else:
                        max_action = random.choice(['h', 's'])

                    if max_action == 's':
                        self.Q1_H[current_state_index] += self.alpha * (reward + self.gamma * self.Q2_S[next_state_index]
                         - self.Q1_H[current_state_index])
                    else:
                        self.Q1_H[current_state_index] += self.alpha * (reward + self.gamma * self.Q2_H[next_state_index]
                         - self.Q1_H[current_state_index])
               
                else: # then we update q2 
                    if (self.Q2_S[next_state_index] > self.Q2_H[next_state_index]):
                        max_action = 's'
                    elif (self.Q2_S[next_state_index] < self.Q2_H[next_state_index]):
                        max_action = 'h'
                    else:
                        max_action = random.choice(['h', 's'])

                    if max_action == 's':
                        self.Q2_H[current_state_index] += self.alpha * (reward + self.gamma * self.Q1_S[next_state_index]
                         - self.Q2_H[current_state_index])
                    else:
                        self.Q2_H[current_state_index] += self.alpha * (reward + self.gamma * self.Q1_H[next_state_index]
                         - self.Q2_H[current_state_index])

            elif action == 's': # if the action was state the we update corresponding Q(State, S) function
                if np.random.rand() < self.beta: # then we update q1 
                    
                    if (self.Q1_S[next_state_index] > self.Q1_H[next_state_index]):
                        max_action = 's'
                    elif (self.Q1_S[next_state_index] < self.Q1_H[next_state_index]):
                        max_action = 'h'
                    else:
                        max_action = random.choice(['h', 's'])

                    if max_action == 's':
                        self.Q1_S[current_state_index] += self.alpha * (reward + self.gamma * self.Q2_S[next_state_index]
                         - self.Q1_S[current_state_index])
                    else:
                        self.Q1_S[current_state_index] += self.alpha * (reward + self.gamma * self.Q2_H[next_state_index]
                         - self.Q1_S[current_state_index])
               
                else: # then we update q2 
                    if (self.Q2_S[next_state_index] > self.Q2_H[next_state_index]):
                        max_action = 's'
                    elif (self.Q2_S[next_state_index] < self.Q2_H[next_state_index]):
                        max_action = 'h'
                    else:
                        max_action = random.choice(['h', 's'])

                    if max_action == 's':
                        self.Q2_S[current_state_index] += self.alpha * (reward + self.gamma * self.Q1_S[next_state_index]
                         - self.Q2_S[current_state_index])
                    else:
                        self.Q2_S[current_state_index] += self.alpha * (reward + self.gamma * self.Q1_H[next_state_index]
                         - self.Q2_S[current_state_index])
            self.update_q()


    def get_Q(self):
        return self.Q_H, self.Q_S
