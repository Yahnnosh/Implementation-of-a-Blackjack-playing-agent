"""
Interface for model-based agents, inherits from agent
"""

from agent import agent
from abc import abstractmethod
import numpy as np

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

    def state_transition_probability(self, initial_state, deck, n=10000):
        """
        Estimates state transition matrix using Monte Carlo method
        :param initial_state: initial state = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck = [n(2), n(3), ..., n(K), n(A)], where n(x) is number of card x remaining
        :param n: number of random sampling for Monte Carlo method
        :return: state transition matrix P
        """
        # Initialization
        N_STATES = 363
        P = np.zeros((N_STATES, N_STATES, 2))   # P[s', s, a], 'hit' = 0, 'stand' = 1
        visits = np.zeros((N_STATES, 2))        # visits[s, a]
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        # Random walks
        for iteration in range(n):
            agent_hand = initial_state[0].copy()
            dealer_hand = [initial_state[1]]
            curr_deck = deck.copy()
            curr_state = [agent_hand.copy(), dealer_hand[0]]
            game_over = False

            while np.random.choice(['hit', 'stand']) == 'hit':  # make random action
                # update visits
                visits[self.state_approx(curr_state), 0] += 1

                # draw a random card
                card = np.random.choice(values, p=(curr_deck / np.sum(curr_deck)))
                curr_deck[values.index(card)] -= 1  # update deck composition
                agent_hand.append(card)
                new_state = [agent_hand.copy(), dealer_hand[0]]

                # check if terminal (busted)
                if self.evaluate(agent_hand) > 21:
                    # update P(s'|s,a)
                    P[1, self.state_approx(curr_state), 0] += 1
                    game_over = True
                    break
                else:
                    # update P(s'|s,a)
                    P[self.state_approx(new_state), self.state_approx(curr_state), 0] += 1
                    curr_state = new_state

            if game_over:
                continue

            # update visits
            visits[self.state_approx(curr_state), 1] += 1

            # predict face down card dealer
            card = np.random.choice(values, p=(curr_deck / np.sum(curr_deck)))
            curr_deck[values.index(card)] -= 1  # update deck composition
            dealer_hand.append(card)

            # let dealer play until the end
            while (self.evaluate(dealer_hand) < 17) or \
                    ((self.evaluate(dealer_hand) == 17) and (self.soft(dealer_hand))):  # S17 rule
                # draw next card
                card = np.random.choice(values, p=(curr_deck / np.sum(curr_deck)))
                curr_deck[values.index(card)] -= 1  # update deck composition
                dealer_hand.append(card)

            # dealer has stopped, check who won
            if (self.evaluate(dealer_hand) > 21) or (self.evaluate(agent_hand) > self.evaluate(dealer_hand)):
                terminal_state = 0  # won
            elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
                terminal_state = 1  # lost
            else:
                terminal_state = 2  # draw
            # update P(s'|s,a)
            P[terminal_state, self.state_approx(curr_state), 1] += 1

        # calculate Monte Carlo estimate
        for state in range(N_STATES):
            for action in (0, 1):
                for target_state in range(N_STATES):
                    if visits[state, action] != 0:
                        P[target_state, state, action] = P[target_state, state, action] / visits[state, action]

        return P
