"""
Agent that acts according to the reference strategy 
-------------------------------------------------------------------------------

"""
from agent import agent
import pandas as pd
 
# converting csv tables to dictionaries
hard_table = pd.read_csv("hard_table.csv", index_col=0).to_dict() # the fixed policy if our hand is hard
soft_table = pd.read_csv("soft_table.csv", index_col=0).to_dict() # the fixed policy if our hand is soft

class table_agent(agent):
    def __init__(self):
        pass


    def policy(self, hand):
        """
        Hits/stands according to the fixed table policy 
        :param hand: hand = [[card1, card2, ..., cardN], card_dealer]
        :return: action
        """
        agent_hand = hand[0] 
        dealer_hand = hand[1]
        
        if dealer_hand in {"J", "Q", "K"}: # they all have the same value
            dealer_hand = "10" 

        agent_sum = self.evaluate(agent_hand) # total card value of the agent 

        if self.soft(agent_hand): 
            action = soft_table[dealer_hand][agent_sum]
        else:
            action = hard_table[dealer_hand][agent_sum]

        return action

    def learn(self, episode):
        pass
