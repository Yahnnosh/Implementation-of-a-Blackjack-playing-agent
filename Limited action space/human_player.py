"""
Interactive interface for Blackjack
"""

from agent import agent

class human_agent(agent):
    def __init__(self):
        pass

    def policy(self, hand):
        """
        Interactive policy for human player
        :param hand: hand = [[card1, card2], card_dealer]
        :return: action (human input)
        """
        print('Your hand:')
        for card in hand[0]:
            print('[', card, ']', end='')
        print('\nDealer face up card:')
        print('[', hand[1], ']')

        action = input('Hit[h] or stand[s]?\n')
        assert ((action == 'h') or (action == 's'))
        return action

    def learn(self, episode):
        pass
