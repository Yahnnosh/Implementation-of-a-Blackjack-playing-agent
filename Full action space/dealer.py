"""
This file serves as the environment (casino)
--------------------------------------------
Requires additional state interpreter
(environment gives agent max available
information, i.e.)
- specific player cards
- specific first face up dealer card
------------------------------------
Blackjack rules implemented:
- 6 decks, deck penetration: 80%
- Dealer hits on soft 17
- Blackjack pays 3:2
- Insurance pays 2:1
- Double down allowed after splitting
- No surrender 
"""

"""
Before implementing anything, let's understand the rules for each action in detail;
these rules vary a lot depending on the particular casino, we will adhere 
(with some exceptions) to the rules as given in the book "Beat the Dealer" by Thorp

1) splitting:
- if first two cards drawn to the player are numerically (!) identical (e.g. King and Jack),
then a player has an option to treat them as the initial cards in two separate hands and 
he receives automatically a second card on each of the split cards. Then he has to place 
additional bet with the amount equail to the initial bet. He then plays his twin hands,
one at a time, as though they were ordinary hand with the following exceptions:
- in case of split aces, player receive only one card on each ace (we ignore this rule for now)
- if a ten valued card falls on one of the split Aces, the hand is counted as ordinary 21 (we ignore this rule for now)
- similarly, if he splits tens and he is given an ace, this counts as ordinary 21 (we ignore this rule for now)
- the player is not allowed to split again (in book, this is different! but one split is much easier to model)
2) doubling down:
- player can double his bet but then he is required to hit and stand 
- double down is allowed after splitting but not allowed if splitted cards are Aces (we ignore this rule for now)
3) insurance:
- player is allowed to place additional bet before the draw if the dealer's card is Ace
- the amount of this additional bet can be any but not higher than the half of the original bet   
- if the dealer has a blackjack, this side bet wins twice its amount 
- the original bet is settled in the usual way, regardless of the side bet
so in case of dealer's blackjack, with indurance there is no net loss or gain 
- we assume that a player cannot buy the insurance after the splitting 
- we assume that this additional bet if always half the original 
4) surrender
- under the Thorpe's rules, there is no such action 
- may be we will include it in our environment later 
"""

"""
TODO:
!Important:
1) Now the agent observes only the total reward at the end of the episode. 
I think it would be more accurate if the agent observes the reward from the insurance separately.  
2) Implement splitting with full rules! 
3) add "learning"

Not important:
1) think of more readable code for "episode['reward'] = reward; agent.learn(episode); return episode" 
"""

"""
Random questions:
1) What about doubling down after insurance? Sounds stupid but is it allowed? 

"""



import random
from table_policy import table_agent


def show(reward, agent_hand, dealer_hand):
    if reward == 0:
        result = 'It\'s a draw!'
    elif reward > 0:
        result = 'You have won!'
    else:
        result = 'You have lost!'

    print('-Game over-\n', result, '\nYour hand:')
    for card in agent_hand:
        print('[', card, ']', end='')
    print('\nDealer hand:')
    for card in dealer_hand:
        print('[', card, ']', end='')
    print('\n')


class dealer:
    def __init__(self):
        self.N_DECKS = 6
        self.PENETRATION = 0.8
        self.VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                       '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
        self.deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * (4 * self.N_DECKS)
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.deck)

    # Checks if more than a (1-`self.PENETRATION`) fraction of `self.deck` has been used. If the latter is true then a
    # new `self.deck` is initialized. Note that it the `agent` is a `count_agent` then we reset its counting.
    def check_deck(self, agent):
        if len(self.deck) < (1 - self.PENETRATION) * 52 * self.N_DECKS:  # check if deck over penetration threshold
            self.deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * (4 * self.N_DECKS)
            self.shuffle()

            #if isinstance(agent, count_agent):
             #   agent.reset_counting()  # we also reset counting (in case of the counting agent)

    # Draws a single card from `self.deck`.
    def draw(self, agent):
        # check if deck needs to be reshuffled for next round   # TODO: legal?
        self.check_deck(agent) # TODO: I think this one is useless 

        return self.deck.pop()

    # Draws two cards from `self.deck`. Note that this function is used in the beginning of the game where both the
    # player and the dealer draw two cards.
    def draw_hand(self, agent):
        # check if deck needs to be reshuffled for next round   # TODO: legal?
        self.check_deck(agent) # TODO: I think this one is useless 

        return [self.draw(agent), self.draw(agent)]

    # Returns True if the value of `hand` is more than 21.
    def busted(self, hand):
        return sum([self.VALUES[card] for card in hand]) > 21

    # Return True if `hand` is a blackjack. A blackjack happens when one of the cards is an ace (`A`) and the second one
    # has a value of 10.
    def blackjack(self, hand):
        return (len(hand) == 2) and (self.evaluate(hand) == 21) 

    # Return True if cards can be splitted 
    def split(self, hand): 
        hand = ['10' if x in ['J', 'Q', 'K'] else x for x in hand] # all face cards are worth ten
        return hand[0] == hand[1]

    # Evaluates and returns the value of `hand`. If an ace is in `hand` then we check if that ace could be used as an
    # 11 and return the sum of the values of the cards + 10.
    def evaluate(self, hand):
        val = sum([self.VALUES[card] for card in hand])
        if 'A' in hand:
            return (val + 10) if (val + 10 <= 21) else val  # only 1 ace in hand can ever be used as 11
        else:
            return val

    # Returns True if `hand` is soft. The latter happens if the hand contains an ace which could be used either as a 1
    # or an 11 without having a value strictly bigger than 21. An equivalent way to check that is to compare the value
    # of the `hand` with the sum of values of the cards in the `hand`
    def soft(self, hand):
        return self.evaluate(hand) - sum([self.VALUES[card] for card in hand]) == 10

    def play_round(self, agent, bet=1, learning=True, splitting=False):
        """
        Plays one round of Blackjack
        :param agent: deterministic agent - needs to implement policy(state)
        :param bet: betting amount
        :return: full episode of a round (hands, actions, reward)
        """
        # Save episode (for agent learning)
        episode = {'hands': [], 'dealer': [], 'actions': [], 'reward': []}  # dealer always shows first card
        allowed_actions = [] # this list contains allowed actions

        # Checks if more than a (1-`self.PENETRATION`) fraction of `self.deck` has been used, in that case we reshuffle.
        self.check_deck(agent)

        # The initial hand is drawn for both the dealer and the agent.
        # If this round is played with the hands that are formed after the splitting then we don't draw the cards here 
        # and just take the stored hands
        agent_hand = self.draw_hand(agent) if (splitting == False) else splitting[0] 
        dealer_hand = self.draw_hand(agent) if (splitting == False) else splitting[1]

        # `episode['hands']` and `episode['dealer']` are updated to include the first two cards of both the agent and
        # the dealer.
        episode['hands'].append(agent_hand.copy())
        episode['dealer'].append(dealer_hand.copy())

        state = [agent_hand, dealer_hand[0]] # the state the agent observes initially 
        reward = 0 # we need initialization for increments 
        
        if (not self.blackjack(agent_hand)): # if the agent does not win automatically
            allowed_actions = ['hit','stand','double'] 
            if (dealer_hand[0] == 'A') and (splitting == False): # agent is not allowed to buy insurance after splitting
                # If the dealer's card is Ace and the player doesn't have a blackjack, the player has an option to place
                # additional bet agaist that the dealer has a blackjack (this side bet is assumed to be half of the initial bet)
                allowed_actions.append('insurance') # agent has an option to place side bet 
            if self.split(agent_hand) and (splitting == False): # if agent can split and if this is the first split 
                allowed_actions.append('split') # then he is allowed to split 
            
            action = agent.policy(state, allowed_actions) # agent's first action
            if self.blackjack(dealer_hand):
                reward = 0  # So, if a dealer hits Blackjack then reward = 0
                episode['reward'] = reward
                if learning:
                    agent.learn(episode)
                return episode
            else:   # otherwise would punish action even though didnt affect outcome    # TODO: remove above if line from all actions below
                episode['actions'].append(action)
        else:
            action = 'no_action' # TODO change 

        if action == 'insurance': # if agent decided to buy the insurance 
            allowed_actions.remove('insurance') # he is not able to buy the insurance again in this round 
            if self.blackjack(dealer_hand):
                reward = 0 # So, if a dealer hits Blackjack then reward = 0 
                episode['reward'] = reward
                if learning:
                    agent.learn(episode)
                return episode
            else: 
                reward = -bet/2 # agent loses the side bet but the game continues 
                action = agent.policy(state, allowed_actions) # agent takes second action
                episode['actions'].append(action)
                return episode
     
        if (action == 'split'): 
            # construct two hands and play them sequentially 
            hand1 = [[agent_hand[0], self.draw(agent)], dealer_hand] 
            hand2 = [[agent_hand[1], self.draw(agent)], dealer_hand] 
        		
            episode1 = self.play_round(agent, bet=bet, splitting=hand1)
            episode2 = self.play_round(agent, bet=bet, splitting=hand2)

            # reward of the splitting episode is just a sum of the rewards of "daughter" hands 
            reward += episode1['reward'] + episode2['reward']
            episode['reward'] = reward
            if learning:
                agent.learn(episode)
            return episode

        if (action == 'double'):     
            bet = 2*bet # the agent doubles his bet 
            if self.blackjack(dealer_hand): # if the dealer has a natural, the agent loses everything
                reward += -bet
                episode['reward'] = reward
                if learning:
                    agent.learn(episode)
                return episode
            else: 
                # we just play usual game after doubling down with the only exception 
                # that we hit only one time

                agent_hand.append(self.draw(agent)) # the agent must hit and stand if doubling down
                episode['hands'].append(agent_hand.copy())
                agent_busted = self.busted(agent_hand) # check if the agent is busted or not 
        			
                if agent_busted:
                    reward += -bet
                    episode['reward'] = reward
                    if learning:
                        agent.learn(episode)
                    return episode 
			
                # next the dealer takes his cards 
                while (self.evaluate(dealer_hand) < 17) or ((self.evaluate(dealer_hand) == 17) and (self.soft(dealer_hand))):
                    dealer_hand.append(self.draw(agent))
                    episode['dealer'].append(dealer_hand.copy())

                # check if the dealer is busted 
                if self.busted(dealer_hand):
                    reward += bet
                    episode['reward'] = reward
                    if learning:
                        agent.learn(episode)
                    return episode 

                # compare values of hands of agent and dealer  
                if self.evaluate(agent_hand) > self.evaluate(dealer_hand):
                    reward += bet
                # If the `agent` lost.
                elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
                    reward += -bet
                # if the game ended as a draw.
                elif self.evaluate(agent_hand) == self.evaluate(dealer_hand):
                    reward += 0
                episode['reward'] = reward
                if learning:
                    agent.learn(episode)
                return episode

        # Check for blackjacks using the following rules:
        # (1) If both the `dealer_hand` and the `agent_hand` are blackjacks then the reward of the agent is 0.
        # (2) If the `dealer_hand` is a blackjack and the `agent_hand` is not a blackjack then the agent loses its bet
        #     and its reward is `-bet`.
        # (3) If the `agent_hand` is a blackjack and the `dealer_hand` is not a blackjack then the agent wins and its
        #     reward is equal to 1.5 * `bet`.
        if self.blackjack(dealer_hand) and self.blackjack(agent_hand):
            reward += 0
        elif self.blackjack(dealer_hand) and (not self.blackjack(agent_hand)):
            reward += -bet
        elif (not self.blackjack(dealer_hand)) and self.blackjack(agent_hand):
            reward += 1.5 * bet

        # If either the `dealer_hand` or the `agent_hand` was a blackjack then the current round of blackjack ends.
        if self.blackjack(dealer_hand) or self.blackjack(agent_hand):
            episode['reward'] = reward
            # The `agent` receives the episodes details in order to learn
            if learning:
                agent.learn(episode)

            # If the `agent` is a `count_agent` then the agent should update its dynamic betting policy.
            # TODO: This should be added as a function when the agent receives the episode
            #if isinstance(agent, count_agent):
             #   agent.dynamic_bet([agent_hand, dealer_hand])

            # (Optional) only for visualization. If the `agent` is a `human_agent` then the details of the current round
            # are visualized.
            #if isinstance(agent, human_agent):
             #   show(reward, agent_hand, dealer_hand)

            return episode

        # The initial `dealer_hand` and `agent_hand` were not blackjacks. The `agent` proceeds playing.

        # The `agent` plays until either `agent_busted` is True or it decides to stand, which happens iff `action` is
        # set to `s`.

        agent_busted = False

        if action == 'hit':
            agent_hand.append(self.draw(agent))  # draw card
            episode['hands'].append(agent_hand.copy())
            agent_busted = self.busted(agent_hand)  # check if busted

        allowed_actions = ['hit','stand'] # only two actions are allowed now   
        
        while (not agent_busted) and (not action == 'stand'):
            # Initialization of the `state` which will be used by the `agent` to decide and update its policies. Note
            # that the agent does not know both the cards of the `dealer` but he has access to only one of them.
            
            state = [agent_hand, dealer_hand[0]]
            action = agent.policy(state, allowed_actions) # agent hits or stands 
            episode['actions'].append(action)

            if action == 'hit':
                agent_hand.append(self.draw(agent))  # draw card
                episode['hands'].append(agent_hand.copy())
                agent_busted = self.busted(agent_hand)  # check if busted

            # TODO: Add comments for what happens if `agent` is a `model_based_agent`.
            #if isinstance(agent, model_based_agent):
            #    cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
            #    concealed_deck = [self.deck.count(value) for value in cards]
            #    concealed_deck[cards.index(dealer_hand[1])] += 1  # add dealer face down card back in
             #   action = agent.policy(state, concealed_deck)
            #else:
                # The `agent` decides which `action` to take based in its current `state`.
            #    action = agent.policy(state)


            # Update `episode['actions']` to include the last action that the player decided to take.
            #episode['actions'].append(action)
            # If the `action` is 'h' then:
            # (1) the `agent` draws a card from the `self.deck`;
            # (2) `episode['hands']` gets updated; and
            # (3) we check if the agent is busted.
            

            #if isinstance(agent, sarsa_agent) and (action == 'h'):  # SARSA is online method => we constantly learn!
               # agent.learn(episode) 

        # If the `agent` is busted then round should end (there is not reason for the `dealer` to play).
        if agent_busted:
            reward += -bet
            #if isinstance(agent, human_agent):
             #   show(reward, agent_hand, dealer_hand)
            #if isinstance(agent, count_agent):
             #   agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
            episode['reward'] = reward
            if learning:
                agent.learn(episode)
            return episode

        # The `dealer` draws a card until one of the following happens:
        # (1) the value of `dealer_hand` is more than 17;
        # (2) the value of `dealer_hand` is 17 and it is not soft.
        # This policy will be called S17 policy.
        while (self.evaluate(dealer_hand) < 17) or ((self.evaluate(dealer_hand) == 17) and (self.soft(dealer_hand))):
            dealer_hand.append(self.draw(agent))
            episode['dealer'].append(dealer_hand.copy())
        # Note that following the previous S17 policy. The dealer may be busted. In that case we know that the `agent`
        # won this round.
        if self.busted(dealer_hand):
            reward += bet
            #if isinstance(agent, human_agent):
             #   show(reward, agent_hand, dealer_hand)
            #if isinstance(agent, count_agent):
             #   agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
            episode['reward'] = reward
            if learning:
                agent.learn(episode)
            return episode

        # At this point we know that both the `agent` and the `dealer` are not busted. Thus we proceed comparing the
        # value of their hands to decide who won and the corresponding reward.
        # If the `agent` won.
        if self.evaluate(agent_hand) > self.evaluate(dealer_hand):
            reward += bet
        # If the `agent` lost.
        elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
            reward += -bet
        # if the game ended as a draw.
        elif self.evaluate(agent_hand) == self.evaluate(dealer_hand):
            reward += 0

       # if isinstance(agent, human_agent):
        #    show(reward, agent_hand, dealer_hand)
        #if isinstance(agent, count_agent):
         #   agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
        episode['reward'] = reward
        if learning:
            agent.learn(episode)
        return episode

    def card_counter(self):
        """
        Returns the number of cards still left in the deck
        :return: [n_2, n_3, n_4, n_5, n_6, n_7, n_8, n_9, n_10', n_J, n_Q, n_K, n_A]
        """
        # check if deck needs to be reshuffled for next round
        self.check_deck(None)

        return [self.deck.count(value) for value in
                ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]


if __name__ == '__main__':
    print('Welcome to Blackjack!\n')

    # Policy selection
    #agent = human_agent()  # interactive agent

    # Play Blackjack
    casino = dealer()

    #while True:
     #   casino.play_round(agent)

      #  if input('Play again? [y][n]') == 'n':
       #     break

