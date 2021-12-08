"""
Interface for agents
"""

from abc import abstractmethod, ABCMeta

class agent(metaclass=ABCMeta):
    @abstractmethod
    def policy(self, hand, allowed_actions):
        """
        Deterministic policy π(s) = a
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :param allowed_actions: [action1, action2, ...]
        :return: action
        """
        pass

    @abstractmethod
    def learn(self, episode):
        """
        Learn from played sequence following π
        :param episode: {'hands': [...], 'dealer': [...], 'actions': [...], 'reward': [...]}
        :return: None
        """
        pass

    # Interpreter:
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

    def soft(self, hand):
        """
        Returns True if hand is soft (one ace counts as 11)
        :param hand: [card1, card2, ..., cardN] where card in ['2', '3', ..., 'K', 'A']
        :return: True if hand is soft else False
        """
        values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                  '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
        return self.evaluate(hand) - sum([values[card] for card in hand]) == 10

    def state_approx(self, hand):
        """
        Approximates state to single number (index of state_approx vector
        (starting from 0!)) - warning: does not assign to terminal states! -
        -------------------------------------------------------------------------
        state_approx = (value(player_hand), value(dealer_hand), soft(player_hand),
                        first_hand(player_hand), pair(player_hand))
        --------------------------------------------------------------------------
        state_approx space = {win, loss, draw,
        [4, 2, 0, 0, 0], [4, 2, 0, 0, 1], [4, 2, 0, 1, 0], [4, 2, 0, 1, 1], ... ,
        [4, 11, 1, 1, 1], ..., [21, 11, 0, 0, 0], ...,  [21, 11, 1, 1, 1]}
        --------------------------------------------------------------------------
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :return index of approximated state in state_approx vector (starting from 0!)
        """
        # TODO: implement for full action space
        raise NotImplementedError
