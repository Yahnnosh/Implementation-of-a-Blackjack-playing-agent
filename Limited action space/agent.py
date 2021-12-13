"""
Interface for agents
"""

from abc import abstractmethod, ABCMeta

class agent(metaclass=ABCMeta):
    @abstractmethod
    def policy(self, hand):
        """
        Deterministic policy π(s) = a
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
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

    def alternative_state_approx(self, hand): # state_approx + the number of cards in your hand
        """
        Approximates state to single number (index of state_approx vector
        (starting from 0!)) - warning: does not assign to terminal states! -
        --------------------------------------------------------------------
        state_approx vector = [win, loss, draw,
        [4, 2, 0], [4, 2, 1], [4, 3, 0], [4, 3, 1], ... ,
        [4, 11, 0], [4, 11, 1], ..., [21, 11, 0], [21, 11, 1]]^T
        where [x, y, a, z] = [sum of values of agent's hand, value of dealer's hand, number of cards in
        the habd, bool(agent hand soft)]
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer, number of cards in the hand]
        :return index of approximated state in state_approx vector (starting from 0!)
        """
        agent_hand = hand[0]
        dealer_hand = hand[1]
        x, y, z = self.evaluate(agent_hand), self.evaluate([dealer_hand]), self.soft(agent_hand)
        return int((3 + (x - 4) * 20) + ((y - 2) * 2) + z)


