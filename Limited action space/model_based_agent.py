"""
Interface for model-based agents, inherits from agent
"""

from agent import agent
from abc import abstractmethod
import numpy as np
import copy

class model_based_agent(agent):
    @abstractmethod
    def policy(self, hand, deck):
        """
        Deterministic policy π(s) = a
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck compostion at current state, dealer face down card included
        :return: action
        """
        pass

    @abstractmethod
    def learn(self, episode):
        pass

    def state_transition_probability(self, initial_state, deck, n=100000):
        """
        Estimates state transition matrix using Monte Carlo method
        :param initial_state: initial state = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck = [n(2), n(3), ..., n(K), n(A)], where n(x) is number of card x remaining
        :param n: number of random sampling for Monte Carlo method
        :return: state transition matrix P
        """
        # Initialization
        N_STATES = 363
        P = np.zeros((N_STATES, N_STATES, 2))     # P[s', s, a], 'hit' = 0, 'stand' = 1
        visits = np.zeros((N_STATES, N_STATES, 2))
        values = np.array(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])

        # Random walks
        for i in range(n):
            old_state = copy.deepcopy(initial_state)   # don't interfere with dealer.py
            old_deck = np.array(deck)
            while True: # do one random walk
                # make random action
                action = np.random.choice(['hit', 'stand'])

                if action == 'hit':
                    # draw a random card
                    card = np.random.choice(values, p=old_deck / np.sum(old_deck))
                    card_index = np.where(values == card)[0][0]

                    # calculate next state
                    next_state = copy.deepcopy(old_state)
                    next_state[0].append(card)
                    agent_hand = next_state[0]

                    # check if terminal (here: busted)
                    if self.evaluate(agent_hand) > 21:
                        # update P(s'|s,a)
                        P[1, self.state_approx(old_state), action == 'stand'] += old_deck[card_index] / np.sum(old_deck)
                        # update visits
                        visits[1, self.state_approx(old_state), action == 'stand'] += 1
                        break
                    else:
                        # update P(s'|s,a)
                        P[self.state_approx(next_state), self.state_approx(old_state), action == 'stand'] += \
                            old_deck[card_index] / np.sum(old_deck)
                        # update visits
                        visits[self.state_approx(next_state), self.state_approx(old_state), action == 'stand'] += 1

                        # update state, deck
                        old_state = next_state
                        old_deck = deck.copy()
                        old_deck[card_index] -= 1
                else:
                    # predict face down card dealer
                    card = np.random.choice(values, p=old_deck / np.sum(old_deck))
                    card_index = np.where(values == card)[0][0]
                    p = old_deck[card_index] / np.sum(old_deck)  # = P(face down card = card)
                    dealer_hand = [old_state[1]]
                    dealer_hand.append(card)

                    # let dealer play until the end
                    while (self.evaluate(dealer_hand) < 17) or \
                            ((self.evaluate(dealer_hand) == 17) and (self.soft(dealer_hand))):  # S17 rule
                        # draw next card
                        card = np.random.choice(values, p=old_deck / np.sum(old_deck))
                        card_index = np.where(values == card)[0][0]
                        p *= old_deck[card_index] / np.sum(old_deck)    # = P(face down card = prev. card, next drawn card = card)
                        dealer_hand.append(card)

                        # check if terminal (here: busted)
                        if self.evaluate(dealer_hand) > 21:
                            # update P(s'|s,a)
                            P[0, self.state_approx(old_state), action == 'stand'] += p
                            # update visits
                            visits[0, self.state_approx(old_state), action == 'stand'] += 1
                            break

                    # dealer has stopped, check who won
                    agent_hand = old_state[0]
                    if self.evaluate(agent_hand) > self.evaluate(dealer_hand):
                        # won
                        terminal_state = 0
                    elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
                        # lost
                        terminal_state = 1
                    else:
                        # draw
                        terminal_state = 2
                    # update P(s'|s,a)
                    P[terminal_state, self.state_approx(old_state), action == 'stand'] += p
                    # update visits
                    visits[terminal_state, self.state_approx(old_state), action == 'stand'] += 1
                    break

        return np.divide(P, visits, where=(visits != 0))
