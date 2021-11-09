"""
Agent that follows the same strategy as the dealer, i.e. draw until 17, hit S17
-------------------------------------------------------------------------------
Serves as medium baseline
"""

from agent import agent

class dealer_policy(agent):
    def __init__(self):
        pass

    def policy(self, hand):
        """
        Hits if hand < 17, hits S17
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :return: action
        """
        agent_hand = hand[0]
        if (self.evaluate(agent_hand) < 17) or ((self.evaluate(agent_hand) == 17) and (self.soft(agent_hand))):
            action = 'h'
        else:
            action = 's'
        return action

    def learn(self, episode):
        pass
