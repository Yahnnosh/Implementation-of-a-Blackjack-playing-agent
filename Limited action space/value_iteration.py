"""
Agent that treats every round of Blackjack as a seperate MDP
and acts according to the policy found by value iteration
"""

from model_based_agent import model_based_agent
import numpy as np

class value_iteration(model_based_agent):
    def __init__(self):
        pass

    def learn(self, episode):
        pass

    def policy(self, hand, deck):
        """
        Does value iteration on current round, return greedy policy on calc. Q functions
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck compostion at current state, dealer face down card included
        :return: action
        """

        # Initialization
        GAMMA = 1  # for episodic tasks makes sense to keep at 1
        N_STATES = 363
        INITIAL_VALUES = -100
        V = INITIAL_VALUES * np.ones(N_STATES)          # V[state]
        '''Q = INITIAL_VALUES * np.ones((N_STATES, 2))     # Q[state, action], 'hit' = 0, 'stand' = 1'''
        R = np.zeros(N_STATES)                          # R[state]
        R[0], R[1], R[2] = 1, -1, 0                     # terminal states
        P = self.state_transition_probability(hand, deck, n=10000)  # P[s', s, a]

        # Value iteration
        for i in range(10):   #TODO: termination criterion
            for state in range(N_STATES):
                max_Q = float('-inf')   # current max
                for action in (0, 1):   # 'hit' = 0, 'stand' = 1
                    curr_Q = R[state]
                    for next_state in range(N_STATES):
                        curr_Q += GAMMA * P[next_state, state, action] * V[next_state]
                    max_Q = curr_Q if curr_Q > max_Q else max_Q
                V[state] = max_Q

        # Greedy policy
        curr_state = self.state_approx(hand)
        # update Q value for current state
        Q = [0, 0]
        for action in (0, 1):  # 'hit' = 0, 'stand' = 1
            Q[action] = R[curr_state]
            for next_state in range(N_STATES):
                Q[action] += GAMMA * P[next_state, curr_state, action] * V[next_state]
        action = np.argmax(Q)

        return 'hit' if action == 0 else 'stand'
