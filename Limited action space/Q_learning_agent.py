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
        dealer_card = episode['dealer'][0][0]
        # TODO: it is possible that `agent_hands` contains exactly one hand, while `actions` is an empty list. This is
        # incorrect. Indeed, in that case `actions` should contain an 's'.
        if not actions:
            return
        if reward > 0:
            final_state_index = 0
        elif reward < 0:
            final_state_index = 1
        elif reward == 0:
            final_state_index = 2
        while len(agent_hands) > 1:
            current_agent_hand = agent_hands.pop(0)
            # compute the next state
            next_agent_hand = agent_hands[0]
            # compute current's and next's state actions
            current_state_index = self.state_approx([current_agent_hand, dealer_card])
            next_state_index = self.state_approx([next_agent_hand, dealer_card])
            # compute the current action
            action = actions.pop(0)
            # If the `next_state_index` is more than ---. then we lost
            if next_state_index >= self.NUMBER_OF_STATES:
                next_state_index = final_state_index
                if not (final_state_index == 1):
                    print("Houston we have a problem!")
            if action == 'h':
                self.H[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.H[
                    current_state_index])
            elif action == 's':
                self.S[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[
                    current_state_index])

        # In this case either actions is empty, which happens iff the last action was a 'h' and we got busted or actions
        # contains exactly one element which is an 's'. We treat each case differently. In the former case we do nothing
        # since the hand that has remained in the `agents_hand` does not correspond to a valid state, but just to a
        # losing state (and we dealt with it inside the while loop). In tha latter case we need to update self.S
        if actions:
            # sanity check
            if not (len(actions) == 1):
                print("problem 1")
            action = actions.pop(0)
            if not (action == 's'):
                print("problem 2")
            # now the computation is repeated
            current_agent_hand = agent_hands.pop(0)
            current_state_index = self.state_approx([current_agent_hand, dealer_card])
            next_state_index = final_state_index
            self.S[current_state_index] += self.alpha * (
                reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[current_state_index])

