"""
Agent that acts according to the reference strategy
"""

from agent import agent
import pandas as pd

# converting csv tables to dictionaries
hard_table = pd.read_csv("hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard
soft_table = pd.read_csv("soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft
split_table = pd.read_csv("split_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is splittable


class table_agent(agent):
    def __init__(self):
        pass

    # TODO: depending on environment implementation need to change this
    def policy(self, hand, allowed_actions):
        """
        Acts according to the fixed table policy
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :param allowed_actions: [action1, action2, ...]
        :return: action
        """
        agent_hand = hand[0]
        dealer_hand = hand[1]

        # translate 10 face values for table use
        if dealer_hand in {"J", "Q", "K"}:
            dealer_hand = "10"

        agent_sum = self.evaluate(agent_hand)  # total card value of the agent

        # check if splittable
        if 'split' in allowed_actions:
            if split_table[dealer_hand][agent_hand[0]]:  # if splitting recommended
                return 'split'

        # check if soft
        if self.soft(agent_hand):
            action = soft_table[dealer_hand][agent_sum]
        else:
            action = hard_table[dealer_hand][agent_sum]

        return action

    def learn(self, episode):
        pass
