"""
Dynamic betting agent acting under the HiLO strategy
"""

from dynamic_betting_agent import Dynamic_betting_agent
import numpy as np

class HiLO(Dynamic_betting_agent):
    def __init__(self, static_betting_policy, min_bet=1, max_bet=100, increment=1, hilo_increment=1):
        """
        Deterministic model-based dynamic betting stratey π(s) = a
        where s = deck before round, a = betting amount
        :param static_betting_policy: augmented static betting policy
        :param min_bet: minimum betting amount
        :param max_bet: maximum betting amount
        :param increment: allowed bet increments
        :param hilo_increment: betting increment per true count (the higher the riskier the agent),
        for hilo_increment=='infty' the betting set collapses to the binary setting (bet_min, bet_max)
        """
        super().__init__(static_betting_policy)

        # legality
        assert min_bet > 0 and max_bet > 0 and increment > 0 and hilo_increment > 0

        # dynamic betting policy params
        self.allowed_bets = [bet for bet in range(min_bet, max_bet, increment)]

        # static betting policy params
        self.N_DECKS = 6    # change this for different decks
        self.hilo_increment = hilo_increment
        self.card_values = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1])
        self.running_count = 0
        self.true_count = 0
        self.remaining_decks = self.N_DECKS
        self.last_deck = [4 * self.N_DECKS for card in range(13)]

    def reset(self):
        """
        Call this when deck is reshuffled
        :return: None
        """
        self.running_count = 0
        self.true_count = 0
        self.remaining_decks = self.N_DECKS
        self.last_deck = [4 * self.N_DECKS for card in range(13)]

    def update(self, deck):
        """
        Updates counts based on played cards in last round
        :param deck: deck before next round, deck = [n_2, n_3, ..., n_K, n_A]
        :return: None
        """
        # calculate played cards in last round
        # TODO: this will fail when deck is shuffled during round (should not happen for large decks)
        deck_before_round = np.array(self.last_deck)
        deck_after_round = np.array(deck)
        played_cards = deck_before_round - deck_after_round

        # check for reshuffle
        if np.sum(deck_after_round) > np.sum(deck_before_round):
            self.reset()
            return

        # update counts
        self.running_count += np.sum(played_cards * self.card_values, dtype=int)
        self.remaining_decks = max(int(np.sum(deck) / 52), 1)    # don't allow 0 deck count
        self.true_count = int(self.running_count / self.remaining_decks)    # round down

    def bet(self, deck):
        """
        Returns optimal bet under the static betting policy
        :param deck: deck before next round, deck = [n_2, n_3, ..., n_K, n_A]
        :return: optimal bet under the static betting policy
        """
        min_bet = self.allowed_bets[0]
        max_bet = self.allowed_bets[-1]

        recommended_bet = self.true_count * self.hilo_increment

        return min(max_bet, max(min_bet, recommended_bet))

