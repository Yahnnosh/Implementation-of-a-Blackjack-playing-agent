from agent import agent
import numpy as np
import random
import time 

class QAgent(agent):

    def __init__(self, alpha=0.01):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.MAX_NUMBER_OF_CARDS = 9
        self.epsilon = 0.2 # GLIE policy parameter
        self.S = np.zeros((self.NUMBER_OF_STATES, self.MAX_NUMBER_OF_CARDS)) # this is Q(State, S)
        self.H = np.zeros((self.NUMBER_OF_STATES, self.MAX_NUMBER_OF_CARDS)) # this is Q(State, H)
        self.S_visitations = np.zeros((self.NUMBER_OF_STATES,self.MAX_NUMBER_OF_CARDS)) 
        self.H_visitations = np.zeros((self.NUMBER_OF_STATES,self.MAX_NUMBER_OF_CARDS))
        self.H[0], self.H[1], self.H[2] = 1, -1, 0 # Q(State,H) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.S[0], self.S[1], self.S[2] = 1, -1, 0 # Q(State,S) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.gamma = 1 # we have a single reward at the end => no need to discount anything 
        self.alpha = alpha # learning rate
        self.rand = 0

    def policy(self, hand):
        return self.old_policy(hand)

    def new_policy(self, hand): # this uses old state approximation 
        # Q-learning is an off-policy TD control method and can work in both online/offline regimes.

        old_state_index = self.state_approx(hand) # we return the current state index
        
        H_total = self.H[old_state_index].sum()
        S_total = self.H[old_state_index].sum() 

        if H_total > S_total: # if Q(State, H) > Q(State, S) then we hit 
            return 'h'
        elif H_total < S_total:
            return 's'
        elif H_total == S_total:
            self.rand += 1 
            '''if (self.rand % 1000 == 0):
                print("rand", self.rand)
                print(H_total)
                print(S_total)'''
            return random.choice(['h', 's'])


    def old_policy(self, hand): # this uses new state approximation 
        # Q-learning is an off-policy TD control method and can work in both online/offline regimes.

        old_state_index = self.state_approx(hand) # we return the current state index
        number_of_cards = len(hand[0])
        state_index = (old_state_index, number_of_cards - 2)

        if self.H[state_index] > self.S[state_index]: # if Q(State, H) > Q(State, S) then we hit 
            return 'h'
        elif self.H[state_index] < self.S[state_index]:
            return 's'
        elif self.H[state_index] == self.S[state_index]:
            if (self.H_visitations[state_index] < 10) or (self.S_visitations[state_index] < 10):
                H_total = self.H[old_state_index].sum()
                S_total = self.H[old_state_index].sum()    
                if H_total > S_total:
                    return 'h'
                if H_total < S_total:
                    return 's'
            self.rand += 1 
            '''print("rand", self.rand)'''
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
            
            current_number_of_cards = len(current_agent_hand)
            if next_agent_hand:
                next_number_of_cards = len(next_agent_hand) 
            else:
                next_number_of_cards = 2

            current_state_index = self.state_approx([current_agent_hand, dealer_card]) # index of the current state 
            next_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index # next state can be either the state 

            current_state_index = (current_state_index, (current_number_of_cards - 2)) # the modified state approximation
            next_state_index = (next_state_index, (next_number_of_cards - 2)) # the modified state approximation 

            #print(current_state_index)
            #print(next_state_index)
            #time.sleep(2)

            # correposponding to the next_agent_hand or it can be the terminal_state 

            action = actions.pop(0) # the action which was done in the current state 
            
            if action == 'h': # if the action was hit the we update corresponding Q(State, H) function
                self.H[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.H[
                    current_state_index])
                self.H_visitations[current_state_index] += 1
            
            elif action == 's': # if the action was state the we update corresponding Q(State, S) function
                self.S[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[
                    current_state_index])
                self.S_visitations[current_state_index] += 1

            # uncomment if we want alpha decay self.alpha = (1/ (1/alpha + 1))
            # example if the current alpha is 1/2 then the new alpha becomes 1/3
            #self.alpha = self.alpha/(1 + self.alpha)

    def get_Q(self):
        return self.H, self.S

    def get_visitations(self, hand):
        state_index = self.state_approx(hand)
        number_of_cards = len(hand[0])
        state_index = (state_index, number_of_cards - 2)
        return self.H_visitations[state_index], self.S_visitations[state_index]

if __name__ == '__main__':
    agent = QAgent()
    print(agent.state_approx([['A', '3'], '4']))
