"""
Model-based dynamic betting policy
(creates new MDP before each round of Blackjack)
------------------------------------------------
WARNING:
- requires value function or q function of the applied static betting policy
- use pretrained static betting policy as inaccurate value functions results
in bad bets
----------------------------------------------------------------------------
FOR NOW ONLY WORKS WITH Q-LEARNING (limited action space)!
"""
# TODO: csv somehow put into
# TODO: different strategies (risky, prop)
# TODO: extend to other model-free

# Import agents
from Q_learning_agent import QAgent
from sarsa_agent import sarsa_agent
from mc_agent import mc_agent

class Model_based_dynamic_betting_policy():
    def __init__(self, static_betting_policy, min_bet=1, max_bet=100, increment=1):
        """
        Deterministic model-based dynamic betting stratey π(s) = a
        where s = deck before round, a = betting amount
        (performance depends on accuracy of V(s) or Q(s,a) of static betting policy)
        :param static_betting_policy: augmented pretrained (highly recommended!) static betting policy
        :param min_bet: minimum betting amount
        :param max_bet: maximum betting amount
        :param increment: allowed bet increments
        """
        # legality
        assert min_bet > 0 and max_bet > 0 and increment > 0  # TODO: what if min_bet = 0? does it even bet?

        # dynamic betting policy params
        self.allowed_bets = [bet for bet in range(min_bet, max_bet, increment)]

        # static betting policy params
        self.static_betting_policy = static_betting_policy
        self.V = self.get_V()

    def bet(self, deck, strategy='proportional'):
        """
        Returns optimal bet under the static betting policy
        (accuracy depends on accuracy of V(s) or Q(s,a) of static betting policy)
        :param deck: deck before next round, deck = [n_2, n_3, ..., n_K, n_A]
        :return: optimal bet under the static betting policy
        """

        expected_return = 0  # expected return of next round under static policy
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}

        # draw first hand
        for i, card1 in enumerate(values):
            # check legality
            if deck[i] <= 0:
                continue

            # update prob, deck
            probability_card1 = deck[i] / sum(deck)
            deck_card1 = deck.copy()
            deck_card1[i] -= 1

            # draw second hand
            for j, card2 in enumerate(values):
                # check legality
                if deck_card1[j] <= 0:
                    continue

                # update prob, deck
                probability_card2 = probability_card1 * deck_card1[j] / sum(deck_card1)
                deck_card2 = deck_card1.copy()
                deck_card2[i] -= 1

                # draw dealer card
                for k, dealer_card in enumerate(values):
                    # check legality
                    if deck_card2[k] <= 0:
                        continue

                    # update prob
                    if sum(deck_card2) == 0:    # TODO: del after debug complete
                        print('\n', deck_card2, deck_card1, deck)
                    probability_hand = probability_card2 * deck_card2[k] / sum(deck_card2)

                    # evaluate expected return under hand
                    expected_return += probability_hand * self.V[card1, card2, dealer_card]

        min_bet = self.allowed_bets[0]
        max_bet = self.allowed_bets[-1]
        if strategy == 'risky':
            recommended_bet = self.allowed_bets[-1] if (expected_return > 0) else self.allowed_bets[0]
        elif strategy == 'proportional':
            recommended_bet = expected_return * max_bet # TODO: <--------------

        return min(max_bet, max(min_bet, recommended_bet))

    def soft(self, hand):
        """
        Returns True if hand is soft (one ace counts as 11)
        :param hand: [card1, card2, ..., cardN] where card in ['2', '3', ..., 'K', 'A']
        :return: True if hand is soft else False
        """
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
        return self.evaluate(hand) - sum([values[card] for card in hand]) == 10

    def evaluate(self, hand):
        """
        Returns value of hand
        :param hand: [card1, card2, ..., cardN] where card in ['2', '3', ..., 'K', 'A']
        :return: Value of hand
        """
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
        val = sum([values[card] for card in hand])
        if 'A' in hand:
            return (val + 10) if (val + 10 <= 21) else val  # only 1 ace in hand can ever be used as 11
        else:
            return val

    def state_approx(self, hand):
        """
        Approximates state to single number (index of state_approx vector
        (starting from 0!)) - warning: does not assign to terminal states! -
        --------------------------------------------------------------------
        state_approx vector = [win, loss, draw,
        [4, 2, 0], [4, 2, 1], [4, 3, 0], [4, 3, 1], ... ,
        [4, 11, 0], [4, 11, 1], ..., [21, 11, 0], [21, 11, 1]]^T
        where [x, y, z] = [sum of values of agent's hand, value of dealer's hand, bool(agent hand soft)]
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :return index of approximated state in state_approx vector (starting from 0!)
        """
        agent_hand = hand[0]
        dealer_hand = hand[1]
        x, y, z = self.evaluate(agent_hand), self.evaluate([dealer_hand]), self.soft(agent_hand)
        return int((3 + (x - 4) * 20) + ((y - 2) * 2) + z)

    def get_V(self):
        """
        Gets value function V(s) for all possible next hands s in S
        :return: V(s)
        """
        # Q function
        if isinstance(self.static_betting_policy, QAgent):
            Q = self.static_betting_policy.get_Q()
            V = {}

            values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                      '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
            # draw first hand
            for card1 in values:
                # draw second hand
                for card2 in values:
                    # draw dealer card
                    for dealer_card in values:
                        # evaluate hand
                        V[card1, card2, dealer_card] = \
                            Q[self.state_approx([[card1, card2], dealer_card])]

            return V

        else:
            raise NotImplementedError
