"""
Model-based dynamic betting policy
(creates new MDP before each round of Blackjack)
------------------------------------------------
WARNING:
- requires value function or Q function of the applied static betting policy
- performance of this agent is highly dependent on the accuracy of the Q or
value function of the static betting policy
----------------------------------------------------------------------------
FOR NOW ONLY WORKS WITH Q-LEARNING (limited action space)!
"""
import matplotlib.pyplot as plt

# Import agents
from dynamic_betting_agent import Dynamic_betting_agent

class Model_based_dynamic_betting_policy(Dynamic_betting_agent):
    def __init__(self, static_betting_policy, min_bet=1, max_bet=100, increment=1, strategy='proportional', risk=0):
        """
        Deterministic model-based dynamic betting stratey π(s) = a
        where s = deck before round, a = betting amount
        (performance depends on accuracy of V(s) or Q(s,a) of static betting policy)
        :param static_betting_policy: augmented static betting policy
        :param min_bet: minimum betting amount
        :param max_bet: maximum betting amount
        :param increment: allowed bet increments
        :param strategy: heuristic for optimal bet, allowed: 'risky', 'proportional', 'proportional_down'
        """
        super().__init__(static_betting_policy)

        # legality
        assert min_bet > 0 and max_bet > 0 and increment > 0

        # dynamic betting policy params
        self.allowed_bets = [bet for bet in range(min_bet, max_bet, increment)]
        self.strategy = strategy
        self.risk = risk

        # static betting policy params
        self.V = None   # requires reset after static policy is trained
        self.recording, self.data = False, []   # for visualizing

    def reset(self):
        """
        Reset value functions if policy changed
        :return: None
        """
        self.V = self.get_V()

    def bet(self, deck):
        """
        Returns optimal bet under the static betting policy
        (accuracy depends on accuracy of V(s) or Q(s,a) of static betting policy)
        :param deck: deck before next round, deck = [n_2, n_3, ..., n_K, n_A]
        :return: optimal bet under the static betting policy
        """

        expected_return = 0  # expected return of next round under static policy
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}

        # 1) Calculate expected return for next round under static betting policy
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
                    probability_hand = probability_card2 * deck_card2[k] / sum(deck_card2)

                    # evaluate expected return under hand
                    # if Blackjack needs explicit reward (as agent hasn't learn from Blackjack)
                    expected_return_hand = 1.5 if (self.evaluate([card1, card2]) == 21) \
                        else self.V[card1, card2, dealer_card]
                    expected_return += probability_hand * expected_return_hand

        # 2) Heuristics for optimal bet based on calculated expected return
        min_bet = self.allowed_bets[0]
        max_bet = self.allowed_bets[-1]

        # TODO: change risk here for encouraging higher bets (more risky play)
        expected_return += self.risk
        if self.recording:  # TODO: record
            self.data.append(expected_return)   # TODO: record

        if self.strategy == 'risky':
            recommended_bet = max_bet if (expected_return > 0) else min_bet  # can adapt threshold
        elif self.strategy == 'proportional':
            recommended_bet = expected_return * max_bet
            # round to next allowed value (up or down)
            if (recommended_bet > min_bet) and (recommended_bet < max_bet):   # otherwise cut to min/max
                # not very efficient but should be ok
                self.allowed_bets.append(recommended_bet)
                self.allowed_bets.sort()
                index = self.allowed_bets.index(recommended_bet)
                # closer to next bigger or next lower?
                distance_to_upper = self.allowed_bets[index + 1] - recommended_bet
                distance_to_lower = recommended_bet - self.allowed_bets[index - 1]
                self.allowed_bets.remove(recommended_bet)
                recommended_bet = self.allowed_bets[index + 1] if distance_to_upper < distance_to_lower \
                    else self.allowed_bets[index - 1]
        elif self.strategy == 'proportional_down':
            recommended_bet = expected_return * max_bet
            # round down to next allowed value (safer?)
            if (recommended_bet > min_bet) and (recommended_bet < max_bet):  # otherwise cut to min/max
                # not very efficient but should be ok
                self.allowed_bets.append(recommended_bet)
                self.allowed_bets.sort()
                index = self.allowed_bets.index(recommended_bet)
                self.allowed_bets.remove(recommended_bet)
                recommended_bet = self.allowed_bets[index - 1]
        else:
            raise ValueError

        return min(max_bet, max(min_bet, recommended_bet))

    def get_V(self):
        """
        Gets value function V(s) for all possible next hands s in S
        :return: V(s)
        """
        # Q function
        Q_hit, Q_stand, Q_split, Q_double = self.static_betting_policy.get_Q()
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
                    splittable = 1 if card1 == card2 else 0
                    state_index = (self.state_approx([[card1, card2], dealer_card]), 1, splittable)    # TODO: only for new_policy
                    V[card1, card2, dealer_card] = max(
                        Q_hit[state_index], Q_stand[state_index], Q_split[state_index], Q_double[state_index])

        # TODO: needs adaptation for full action space
        # normalize into -1 < V(s) < +1
        V_min = min(V.values())
        V_max = max(V.values())

        normalize = lambda x: -1 + (x - V_min) / (V_max - V_min) * 2
        for hand in V:
            V[hand] = normalize(V[hand])

        return V

    def record(self):
        self.recording = True

    def show_record(self, ax):
        ax.plot(self.data)
