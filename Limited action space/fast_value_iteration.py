"""
Agent that treats every round of Blackjack as a separate MDP
and acts as a greedy policy on the value functions found by VI
--------------------------------------------------------------
Faster than normal value iteration due to state space cutting
"""

from model_based_agent import model_based_agent
import numpy as np

class fast_value_iteration(model_based_agent):
    def __init__(self):
        pass

    def learn(self, episode):
        pass

    # override
    def state_transition_probability(self, initial_state, deck, n=10000):
        """
        Estimates state transition matrix using Monte Carlo method (but faster by cutting state space)
        :param initial_state: initial state = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck = [n(2), n(3), ..., n(K), n(A)], where n(x) is number of card x remaining
        :param n: number of random sampling for Monte Carlo method
        :return: state transition matrix P
        """
        # Initialization
        P = {}          # P[s][a][s'], 'hit' = 0, 'stand' = 1
        visits = {}     # visits[s][a]
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
                s = self.state_approx(curr_state)
                action = 0
                if s not in visits:     # never started from s
                    visits[s] = {action: 1}
                elif action not in visits[s]:  # started from s but never done action (0: hit, 1: stand)
                    visits[s][action] = 1
                else:   # started from s and already done action
                    visits[s][action] += 1

                # draw a random card
                card = np.random.choice(values, p=(curr_deck / np.sum(curr_deck)))
                curr_deck[values.index(card)] -= 1  # update deck composition
                agent_hand.append(card)
                new_state = [agent_hand.copy(), dealer_hand[0]]

                # check if terminal (busted)
                if self.evaluate(agent_hand) > 21:
                    # update P(s'|s,a)
                    s_new = 1
                    if s not in P:  # never started from s
                        P[s] = {action: {s_new: 1}}
                    elif action not in P[s]:  # started from s but never done action (0: hit, 1: stand)
                        P[s][action] = {s_new: 1}
                    elif s_new not in P[s][action]:  # started from s, already done action but never reached s_new
                        P[s][action][s_new] = 1
                    else:  # started from s, already done action and reached s_new
                        P[s][action][s_new] += 1
                    game_over = True
                    break
                else:
                    # update P(s'|s,a)
                    s_new = self.state_approx(new_state)
                    if s not in P:  # never started from s
                        P[s] = {action: {s_new: 1}}
                    elif action not in P[s]:  # started from s but never done action (0: hit, 1: stand)
                        P[s][action] = {s_new: 1}
                    elif s_new not in P[s][action]:  # started from s, already done action but never reached s_new
                        P[s][action][s_new] = 1
                    else:  # started from s, already done action and reached s_new
                        P[s][action][s_new] += 1
                    curr_state = new_state

            if game_over:
                continue

            # update visits
            s = self.state_approx(curr_state)
            action = 1
            if s not in visits:  # never started from s
                visits[s] = {action: 1}
            elif action not in visits[s]:  # started from s but never done action (0: hit, 1: stand)
                visits[s][action] = 1
            else:  # started from s and already done action
                visits[s][action] += 1

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
            s_new = terminal_state
            if s not in P:  # never started from s
                P[s] = {action: {s_new: 1}}
            elif action not in P[s]:  # started from s but never done action (0: hit, 1: stand)
                P[s][action] = {s_new: 1}
            elif s_new not in P[s][action]:  # started from s, already done action but never reached s_new
                P[s][action][s_new] = 1
            else:  # started from s, already done action and reached s_new
                P[s][action][s_new] += 1

        # calculate Monte Carlo estimate
        for s in P:
            for action in P[s]:
                for s_new in P[s][action]:
                    if visits[s][action] != 0:
                        P[s][action][s_new] /= visits[s][action]

        return P

    def policy(self, hand, deck, iterations=1000):
        """
        Does value iteration on current round, return greedy policy on calc. Q functions
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :param deck: deck compostion at current state, dealer face down card included
        :param iterations: number of iterations for VI
        :return: action
        """
        # Initialization
        GAMMA = 1  # for episodic tasks makes sense to keep at 1
        INITIAL_VALUES = -100
        V = {}  # V[state]
        R = {}  # R[state]
        V[0], V[1], V[2] = 1, -1, 0     # terminal states

        P = self.state_transition_probability(hand, deck, n=10000)  # P[s][a][s']

        # Value iteration
        for i in range(iterations):
            for state in P:
                max_Q = INITIAL_VALUES
                for action in P[state]:
                    Q = 0
                    for next_state in P[state][action]:
                        if next_state in V:
                            Q += GAMMA * P[state][action][next_state] * V[next_state]
                        else:
                            Q += GAMMA * P[state][action][next_state] * INITIAL_VALUES
                    max_Q = Q if Q > max_Q else max_Q
                V[state] = max_Q

        # Greedy policy
        state = self.state_approx(hand)     # current state
        Q = {state: {0: INITIAL_VALUES, 1: INITIAL_VALUES}}    # Q[s][a]
        for action in P[state]:
            Q[state][action] = 0
            for next_state in P[state][action]:
                Q[state][action] += GAMMA * P[state][action][next_state] * V[next_state]
        action = np.argmax([Q[state][0], Q[state][1]])

        return 'h' if action == 0 else 's'
