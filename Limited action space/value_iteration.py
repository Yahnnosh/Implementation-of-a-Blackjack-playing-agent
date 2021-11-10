"""
Agent that treats every round of Blackjack as a separate MDP
and acts as a greedy policy on the value functions found by VI
"""

from model_based_agent import model_based_agent
import numpy as np
import time

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
        R = np.zeros(N_STATES)                          # R[state]
        R[0], R[1], R[2] = 1, -1, 0                     # terminal states

        t0 = time.time()
        P = self.state_transition_probability(hand, deck, n=10000)  # P[s', s, a]
        print('Time for P: ', round(time.time() - t0, 2), 's')

        # Value iteration
        t0 = time.time()
        for i in range(1000):   #TODO: termination criterion
            for state in range(N_STATES):
                V[state] = np.max([R[state] + GAMMA * np.dot(P[:, state, action], V) for action in (0, 1)])
        print('Time for VI: ', round(time.time() - t0, 2), 's')

        # Greedy policy
        curr_state = self.state_approx(hand)
        action = np.argmax([R[curr_state] + GAMMA * np.dot(P[:, curr_state, action], V) for action in (0, 1)])

        return 'hit' if action == 0 else 'stand'
