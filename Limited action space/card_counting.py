"""
Agent that counts cards
-------------------------------------------------------------------------------
the agent tracks the total value of cards played ("running count")
based on which he decides how much he bets in the next round

Here we use Hi_Lo counting strategy. It is not the best but it is the simplest.
Its main advantage is that it is the same regardless of the number of decks used.
Here's how cards are counted: 
{2 - 6} +1
{7 - 9} +0
{10 - A} -1

in pseudo-code it would be something like that 
game
{
    each round (until decks are depleted)
    {
        bet: always minimum until (true_count - 1) > 0 - then #(true_count - 1) of minimum bets
        usual round using "table_policy"
        update # of the decks left
        update running count
        culculate the true count = running count / number of the decks left
        return true count 
    }
    running count = 0 # reset
    number of the decks left = 6 # reset
}
"""

from agent import agent
import pandas as pd
import math
 
# converting csv tables to dictionaries
hard_table = pd.read_csv("hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard
soft_table = pd.read_csv("soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft
split_table = pd.read_csv("split_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is splittable

class count_agent(agent):
    def __init__(self):
        self.TOTAL_DECKS = 6 # total number of the decks in the game 
        self.reset_counting()

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

    def count_cards(self, hand):  # returns the running count (it is called after each round of the game)
        hi_lo = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0,
                 '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}  # counting rules
        val = sum([hi_lo[card] for card in hand])  # perform counting of cards played in this round
        self.running_count = self.running_count + val
        return self.running_count

    def count_decks(self, hand):  # tracks how many full decks have not been dealt yet
        self.total_cards = self.total_cards - len(hand)  # tracks how many cards are left in the deck
        self.decks_left = self.total_cards / 52  # in terms of decks
        self.decks_left = math.floor(self.decks_left)  # just rounding down
        if self.decks_left == 0:
             self.decks_left = self.decks_left + 1  # we don't want to divide by zero
        return self.decks_left

    def dynamic_bet(self, hand):  # returns the optimal bet for the next round based on the running count
        hand = [item for sublist in hand for item in sublist]  # just flat list that contains all the cards which were played this round
        decks_left = self.count_decks(hand)
        running_count = self.count_cards(hand)
        true_count = running_count / decks_left  # true count value
        true_count = round(true_count) 
        if (true_count - 1) > 0:
            self.bet = true_count - 1  # bet for the next round depends on the true_count
        else:
            self.bet = 1 # otherwise make minimum bet

    def reset_counting(self):  # we reset when cards get reshuffled
        self.total_cards = self.TOTAL_DECKS * 52
        self.running_count = 0
        self.decks_left = self.TOTAL_DECKS  # the number of FULL decks left; after the first deal there are 5 full decks left
        self.bet = 1  # this is the bet for the next round in terms of the minimum bets
