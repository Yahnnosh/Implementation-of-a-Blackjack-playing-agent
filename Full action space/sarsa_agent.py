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
import pandas as pd

# converting csv tables to dictionaries
double_hard_table = pd.read_csv("double_hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard and double is allowed
double_soft_table = pd.read_csv("double_soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft and double is allowed
hard_table = pd.read_csv("hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard and double is not allowed
soft_table = pd.read_csv("soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft and double is not allowed
split_table = pd.read_csv("split_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is splittable


class Sarsa_agent(agent):

    def __init__(self, alpha=0.01):
        self.NUMBER_OF_STATES = 363  # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)

        # Q-values
        self.Stand = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Stand)
        self.Hit = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Hit)
        self.Split = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Split)
        self.Double = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Doubling)

        # Q-values of terminal states
        self.Stand[0], self.Stand[1], self.Stand[2] = 1, -1, 0
        self.Hit[0], self.Hit[1], self.Hit[2] = 1, -1, 0
        self.Split[0], self.Split[1], self.Split[2] = 1, -1, 0
        self.Double[0], self.Double[1], self.Double[2] = 1, -1, 0

        self.gamma = 1  # we have a single reward at the end => no need to discount anything
        self.alpha = alpha  # learning rate

    def policy(self, hand, allowed_actions):  # given hand and allowed actions, take a certain action
        agent_hand = hand[0]
        dealer_hand = hand[1]

        # translate 10 face values for table use
        if dealer_hand in {"J", "Q", "K"}:
            dealer_hand = "10"

        agent_sum = self.evaluate(agent_hand)  # total card value of the agent

        # check if splittable
        if 'split' in allowed_actions:
            if agent_hand[0] in {"J", "Q", "K"}:
                agent_hand[0] = "10"
            if split_table[dealer_hand][agent_hand[0]]:  # if splitting recommended
                return 'split'

        if 'double' in allowed_actions:
            if self.soft(agent_hand):
                action = double_soft_table[dealer_hand][agent_sum]
            else:
                action = double_hard_table[dealer_hand][agent_sum]
        else:
            if self.soft(agent_hand):
                action = soft_table[dealer_hand][agent_sum]
            else:
                action = hard_table[dealer_hand][agent_sum]

        actions = {
            's': 'stand',
            'h': 'hit',
            'd': 'double'
        }

        action = actions[action]

        return action

    def learn(self, episode):
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions:  # actions list can be empty if either agent or dealer has a blackjack => nothing to learn here, just return
            return

        if len(actions) != len(agent_hands):  # then and only then the agent busted
            del agent_hands[
                -1]  # so we remove the last hand of the agent from the episode because it doesn't carry any useful information (it is just lose state)

        if reward > 0:  # if the reward was positive, we won
            final_state_index = 0  # the index of the terminal win state
        elif reward < 0:
            final_state_index = 1  # the index of the terminal lose state
        elif reward == 0:
            final_state_index = 2  # the index of the terminal draw state

        while agent_hands:  # while there is something we can learn from
            # current state, next state
            current_agent_hand = agent_hands.pop(0)  # current hand
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand

            # state approx: number of cards in hand
            current_number_of_cards = len(current_agent_hand)
            next_number_of_cards = len(next_agent_hand) if next_agent_hand else 2

            # state approx: splittable hand
            current_splittable = 1 if self.split(current_agent_hand) else 0
            next_splittable = 0  # next state is never splittable

            # old state approx
            old_current_state_index = self.state_approx([current_agent_hand, dealer_card])
            old_next_state_index = self.state_approx(
                [next_agent_hand, dealer_card]) if agent_hands else final_state_index

            # new state approximation
            first_hand = lambda x: int(x == 2)
            current_state_index = (old_current_state_index, first_hand(current_number_of_cards), current_splittable)
            next_state_index = (old_next_state_index, first_hand(next_number_of_cards), next_splittable)

            action = actions.pop(0)  # the action which was done in the current state
            Q_max = max(self.Hit[next_state_index], self.Stand[next_state_index])  # the max Q value for the next state

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

    def split(self, hand):
        hand = ['10' if x in ['J', 'Q', 'K'] else x for x in hand]  # all face cards are worth ten
        return hand[0] == hand[1]

