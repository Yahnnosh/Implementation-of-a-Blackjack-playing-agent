# SARSA is on-policy TD control method that resembles policy iteration 
# it is always online, meaning that the policy updates after each action, meaning that there are multiple updates even in a single episode 
# That is in contrast to off-policy Q learning method that updates state action value function only after the whole episode ends! 

# How should we act in the episode under SARSA method?
# 1) random initialization of Q(s,a) for all states and actions 
# 2) observe initial state "s" and choose action "a" according to the greedy policy on Q function from the first step
# Loop:
# 3) take action "a" and observe the next state "s_next" 
# 4) then we choose action "a_next" from "s_next" using greedy-policy from Q
# 5) update Q function as Q(s,a) <- Q(s,a) + alpha[reward + gamma * Q(s_next, a_next) - Q(s, a)]   
# 6) update s <- s_next and a <- a_next 

# so the tricky thing we should remember is that in 3) we act according to the greedy policy based on the previous estimate of the state-action value function 
# and not based on the current estimate.

# in other words: we take first action in an episode based on the most recent estimate of q function  
# but later it is based on the previous estimate

# therefore, SARSA must understand when the episode (round) ends
# the end of an episode can be determined from the reward variable; if it is zero list, then it is not yet the end of the episode
# but when the reward is 0/1/-1 integer, then it reveals that we are at the end of the episode 

# the win rate is around 42.5% (vs 43.5% optimal)

from agent import agent
import numpy as np
import random

class sarsa_agent(agent):

    def __init__(self):
        self.NUMBER_OF_STATES = 363 # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.S = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, S)
        self.H = np.zeros(self.NUMBER_OF_STATES) # this is Q(State, H)
        self.H[0], self.H[1], self.H[2] = 0, 0, 0 # state action value functions of the terminal states are always zero under SARSA algorithm 
        self.S[0], self.S[1], self.S[2] = 0, 0, 0 # state action value functions of the terminal states are always zero under SARSA algorithm 
        self.gamma = 1 # we have a single reward at the end => no need to discount anything 
        self.alpha = 0.01 # learning rate; TODO: 1) perform hyperparameter optimization; 2) see what happens if alpha decays in time ->  
        # -> to guarantee convergence of the state-action value function; now it doesn't converge to 43.5% win rate of the optimal (table) policy  
        self.act_based_on_previous_q = False # this variable decides if we act based on previous (or current) state action value function 
        self.stored_action = '' # this is the action for the current step which is determined from the previous estimate of the q function 

    def greedy_policy(self, hand): # usual greedy policy; it returns the action based on the current estimate of the state-action value function
        state_index = self.state_approx(hand) # we return the current state index
        if self.H[state_index] > self.S[state_index]: # if Q(State, H) > Q(State, S) then we hit 
            action = 'h'
        elif self.H[state_index] < self.S[state_index]:
            action = 's'
        elif self.H[state_index] == self.S[state_index]:
            action = random.choice(['h', 's'])
        return action 

    def policy(self, hand, evaluating = False):
        if evaluating == True:
            self.act_based_on_previous_q = False
        if self.act_based_on_previous_q:
            return self.stored_action # then we act according to the q function from the previous iteration 
        else:
            self.act_based_on_previous_q = True # after we will act based on the previous q 
            return self.greedy_policy(hand) # but now we act according to the most recent q function

    def learn(self, episode):
        #print("learning starts")
        #print(episode)
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions: # actions list can be empty if either agent or dealer has a blackjack => nothing to learn here, just return 
            return

        if not isinstance(reward, int): # if reward = [] that means that the episode has not finished yet 
            reward = 0 # we didn't observe the reward yet = 0 reward
            if self.evaluate(episode['hands'][-1]) > 21: # we don't learn if the agent was busted but the reward has not been yet observed 
                return
        else:
            self.act_based_on_previous_q = False # means that the episode ends; in a new episode we will act based on the most recent update of the q function 

            if len(actions) != len(agent_hands): # it holds when and only when the agent busted 
                del agent_hands[-1] # so we remove the last hand of the agent from the episode because it doesn't carry any useful information for us (it is just lose state)

            if reward > 0: # if the reward was positive, we won
                final_state_index = 0 # the index of the terminal win state
            elif reward < 0:
                final_state_index = 1 # the index of the terminal lose state
            elif reward == 0:
                final_state_index = 2 # the index of the terminal draw state  
       
        current_agent_hand = agent_hands.pop(0) # current agent hand 
        next_agent_hand = agent_hands[0] if agent_hands else None # next agent hand

        next_state = [next_agent_hand, dealer_card] # next state
            
        current_state_index = self.state_approx([current_agent_hand, dealer_card]) # index of the current state 
        next_state_index = self.state_approx(next_state) if agent_hands else final_state_index # the next state can be either the state 
        # correposponding to the next_agent_hand or it can be the terminal_state 

        action = actions.pop(0) # the action which was done in the current state      
        self.stored_action = self.greedy_policy(next_state) if agent_hands else None # the action we take in the next state (if the next state is from this episode)
            
        # updated of state action value functions:    
        if action == 'h': # if the action was hit the we update corresponding Q(State, H) function
            self.H[current_state_index] += self.alpha * (reward + self.gamma * self.H[next_state_index] - self.H[current_state_index])
            
        elif action == 's': # if the action was state the we update corresponding Q(State, S) function
            self.S[current_state_index] += self.alpha * (reward + self.gamma * self.S[next_state_index] - self.S[current_state_index])
