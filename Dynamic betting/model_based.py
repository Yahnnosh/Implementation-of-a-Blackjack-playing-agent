"""
Model-based dynamic betting policy
(creates new MDP before each round of Blackjack)
------------------------------------------------
WARNING:
- requires value function or q function of the applied static betting policy
- highly recommended to use pretrained static betting policy
----------------------------------------------------------------------------
FOR NOW ONLY WORKS WITH Q-LEARNING (limited action space)!
"""

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
        assert(min_bet > 0) # TODO: what if min_bet = 0? does it even try?

        # static betting policy params
        self.static_betting_policy = static_betting_policy
        self.V_static = # TODO: could be changed to be updatable but not desirable

        # dynamic betting policy params
        self.Q = None
        self.P = None
        self.allowed_bets = [bet for bet in range(min_bet, max_bet, increment)]

    def bet(self, deck):
        """
        Returns optimal bet under the static betting policy
        (accuracy depends on accuracy of V(s) or Q(s,a) of static betting policy)
        :param deck: deck before the round, deck = [n_2, n_3, ..., n_K, n_A]
        :return: optimal bet under the static betting policy
        """
        # Build state space S

        # Calculcate P(s'|s, a)

        # Calculate Q(s,a)

        # Greedy policy on Q(s,a)
