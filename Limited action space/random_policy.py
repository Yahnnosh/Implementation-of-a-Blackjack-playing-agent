"""
Agent that does random things
-------------------------------------------------------------------------------
Serves as lowest baseline, win rate is expected to be around 28% 
"""

import random
from agent import agent

class random_agent(agent):
    def __init__(self):
        pass

    def policy(self, hand):
        """
        Hits if hand < 17, hits S17
        :param hand: hand = [[card1, card2], card_dealer]
        :return: action
        """
        action = random.choice(["h","s"]) 
        return action

    def learn(self, episode):
        pass
