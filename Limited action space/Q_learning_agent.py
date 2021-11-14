"""
Interface for model-based agents, inherits from agent
"""

from agent import agent
import numpy as np
import random


class QAgent(agent):

    def __init__(self):
        self.NUMBER_OF_STATES = 363
        # The first three states are considered Win, Lose, Tie
        # TODO: H(state) denotes Q(state, 'h') and S(state) denotes Q(state, 's').
        self.S = np.zeros(self.NUMBER_OF_STATES)
        self.H = np.zeros(self.NUMBER_OF_STATES)
        self.H[0], self.H[1], self.H[2] = 1, -1, 0
        self.S[0], self.S[1], self.S[2] = 1, -1, 0
        self.gamma = 1
        self.alpha = 0.1

    def policy(self, hand):
        """
        Deterministic policy π(s) = a
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :return: action
        """
        state_index = self.state_approx(hand)
        if self.H[state_index] > self.S[state_index]:
            return 'h'
        elif self.H[state_index] < self.S[state_index]:
            return 's'
        elif self.H[state_index] == self.S[state_index]:
            return random.choice(['h', 's'])

    def learn(self, episode):
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']

        if reward > 0:
            final_state_index = 0
        elif reward < 0:
            final_state_index = 1
        elif reward == 0:
            final_state_index = 2
        dealer_card = episode['dealer'][0][0]
        current_agent_hand = agent_hands.pop(0)
        while agent_hands:
            # compute the next state
            next_agent_hand = agent_hands[0]

            # compute current's and next's state actions
            current_state_index = self.state_approx([current_agent_hand, dealer_card])
            next_state_index = self.state_approx([next_agent_hand, dealer_card])
            if next_state_index >= self.NUMBER_OF_STATES or len(agent_hands) == 1:
                next_state_index = final_state_index
            # compute the current action
            action = actions.pop(0)

            if action == 'h':
                self.H[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.H[
                    current_state_index])
            elif action == 's':
                self.S[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[
                    current_state_index])

            # prepare for the next iteration
            #self.alpha = 1.0 / ((1.0 / self.alpha) + 0.2)
            current_agent_hand = agent_hands.pop(0)
