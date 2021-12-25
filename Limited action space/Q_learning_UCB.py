from agent import agent
import numpy as np
import random
import time


class QAgent_UCB(agent):

    def __init__(self, alpha=0.01, UCB_hyperparam=0.5):
        self.NUMBER_OF_STATES = 363  # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)
        self.S = np.zeros(self.NUMBER_OF_STATES)  # this is Q(State, S)
        self.H = np.zeros(self.NUMBER_OF_STATES)  # this is Q(State, H)
        self.S_visitations = np.zeros(self.NUMBER_OF_STATES)
        self.H_visitations = np.zeros(self.NUMBER_OF_STATES)
        self.H[0], self.H[1], self.H[2] =\
            1, -1, 0  # Q(State,H) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.S[0], self.S[1], self.S[2] =\
            1, -1, 0  # Q(State,S) when State is terminal (win/lose/draw); what would happen if we set them to zero?
        self.gamma = 1  # we have a single reward at the end => no need to discount anything
        self.alpha = alpha  # learning rate
        self.UCB_hyperparam = UCB_hyperparam

    def policy(self, hand):  # this uses old state approximation
        # Q-learning is an off-policy TD control method and can work in both online/offline regimes.

        state_index = self.state_approx(hand)  # we return the current state index

        state_visitations = max(1, self.H_visitations[state_index] + self.S_visitations[state_index])
        hit_visitations = max(1, self.H_visitations[state_index])
        stand_visitations = max(1, self.S_visitations[state_index])

        Q_hit = self.H[state_index] + self.UCB_hyperparam * (np.log(state_visitations) / hit_visitations)**0.5
        Q_hit = min(1, Q_hit) if Q_hit > 0 else max(-1, Q_hit) # TODO: check if useful
        Q_stand = self.S[state_index] + self.UCB_hyperparam * (np.log(state_visitations) / stand_visitations)**0.5
        Q_stand = min(1, Q_stand) if Q_stand > 0 else max(-1, Q_stand) # TODO: check if useful

        if (stand_visitations >= 1000) and (hit_visitations >= 3):  # TODO: away for debugging
            a = 1    # TODO: away for debugging
        if Q_hit == Q_stand:
            return random.choice(['h', 's'])
        else:
            return 'h' if Q_hit > Q_stand else 's'

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

            current_agent_hand = agent_hands.pop(0)  # current hand
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand

            current_number_of_cards = len(current_agent_hand)
            if next_agent_hand:
                next_number_of_cards = len(next_agent_hand)
            else:
                next_number_of_cards = 2

            current_state_index = self.state_approx([current_agent_hand, dealer_card])  # index of the current state
            next_state_index = self.state_approx([next_agent_hand,
                                                  dealer_card]) if agent_hands else final_state_index  # next state can be either the state

            # print(current_state_index)
            # print(next_state_index)
            # time.sleep(2)

            # correposponding to the next_agent_hand or it can be the terminal_state

            action = actions.pop(0)  # the action which was done in the current state

            if action == 'h':  # if the action was hit the we update corresponding Q(State, H) function
                self.H[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.H[
                    current_state_index])
                value = self.H[current_state_index]
                self.H[current_state_index] = min(1, value) if value > 0 else max(-1, value) # TODO: check if useful
                self.H_visitations[current_state_index] += 1

            elif action == 's':  # if the action was state the we update corresponding Q(State, S) function
                self.S[current_state_index] += self.alpha * (
                        reward + self.gamma * max(self.H[next_state_index], self.S[next_state_index]) - self.S[
                    current_state_index])
                value = self.S[current_state_index]
                self.S[current_state_index] = min(1, value) if value > 0 else max(-1, value) # TODO: check if useful
                self.S_visitations[current_state_index] += 1

            # uncomment if we want alpha decay self.alpha = (1/ (1/alpha + 1))
            # example if the current alpha is 1/2 then the new alpha becomes 1/3
            # self.alpha = self.alpha/(1 + self.alpha)

    def get_Q(self):
        return self.H, self.S

    def get_Q_hand(self, hand):
        state_index = self.state_approx(hand)
        return self.H[state_index], self.S[state_index]

    def get_visitations(self, hand):
        state_index = self.state_approx(hand)
        return self.H_visitations[state_index], self.S_visitations[state_index]
