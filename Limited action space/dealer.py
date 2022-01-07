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

import random
from human_player import human_agent
from random_policy import random_agent
from table_policy import table_agent
from card_counting import count_agent
from model_based_agent import model_based_agent
from value_iteration import value_iteration
from sarsa_agent_old import sarsa_agent
from sarsa_agent import SARSA_agent


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
    def __init__(self, decks=6, penetration=0.8, infinity=False):
        self.N_DECKS = decks
        self.PENETRATION = penetration
        self.VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                       '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10, 'A': 1}
        self.deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * (4 * self.N_DECKS)
        self.shuffle()
        self.infinity = infinity  # simulates infinite deck

    def shuffle(self):
        random.shuffle(self.deck)

    # Checks if more than a (1-`self.PENETRATION`) fraction of `self.deck` has been used. If the latter is true then a
    # new `self.deck` is initialized. Note that it the `agent` is a `count_agent` then we reset its counting.
    def check_deck(self, agent):
        if len(self.deck) < (1 - self.PENETRATION) * 52 * self.N_DECKS:  # check if deck over penetration threshold
            self.deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * (4 * self.N_DECKS)
            self.shuffle()

            if isinstance(agent, count_agent):
                agent.reset_counting()  # we also reset counting (in case of the counting agent)

    # Draws a single card from `self.deck`.
    def draw(self, agent):
        if self.infinity:
            return random.choice(['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'])

        # check if deck needs to be reshuffled for next round   # TODO: legal?
        self.check_deck(agent)

        return self.deck.pop()

    # Draws two cards from `self.deck`. Note that this function is used in the beginning of the game where both the
    # player and the dealer draw two cards.
    def draw_hand(self, agent):
        # check if deck needs to be reshuffled for next round   # TODO: legal?
        self.check_deck(agent)

        return [self.draw(agent), self.draw(agent)]

    # Returns True if the value of `hand` is more than 21.
    def busted(self, hand):
        return sum([self.VALUES[card] for card in hand]) > 21

    # Return True if `hand` is a blackjack. A blackjack happens when one of the cards is an ace (`A`) and the second one
    # has a value of 10.
    def blackjack(self, hand):
        return (len(hand) == 2) and (self.evaluate(hand) == 21) 

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

    def play_round(self, agent, bet=1, learning=True):
        """
        Plays one round of Blackjack
        :param agent: deterministic agent - needs to implement policy(state)
        :param bet: betting amount
        :return: full episode of a round (hands, actions, reward)
        """
        # Save episode (for agent learning)
        episode = {'hands': [], 'dealer': [], 'actions': [], 'reward': []}  # dealer always shows first card

        # Checks if more than a (1-`self.PENETRATION`) fraction of `self.deck` has been used, in that case we reshuffle.
        self.check_deck(agent)

        # The initial hand is draw for both the dealer and the agent.
        agent_hand = self.draw_hand(agent)
        dealer_hand = self.draw_hand(agent)

        # `episode['hands']` and `episode['dealer']` are updated to include the first two cards of both the agent and
        # the dealer.
        episode['hands'].append(agent_hand.copy())
        episode['dealer'].append(dealer_hand.copy())

        # Check for blackjacks using the following rules:
        # (1) If both the `dealer_hand` and the `agent_hand` are blackjacks then the reward of the agent is 0.
        # (2) If the `dealer_hand` is a blackjack and the `agent_hand` is not a blackjack then the agent loses its bet
        #     and its reward is `-bet`.
        # (3) If the `agent_hand` is a blackjack and the `dealer_hand` is not a blackjack then the agent wins and its
        #     reward is equal to 1.5 * `bet`.
        if self.blackjack(dealer_hand) and self.blackjack(agent_hand):
            reward = 0
        elif self.blackjack(dealer_hand) and (not self.blackjack(agent_hand)):
            reward = -bet
        elif (not self.blackjack(dealer_hand)) and self.blackjack(agent_hand):
            reward = 1.5 * bet

        # If either the `dealer_hand` or the `agent_hand` was a blackjack then the current round of blackjack ends.
        if self.blackjack(dealer_hand) or self.blackjack(agent_hand):
            episode['reward'] = reward
            # The `agent` receives the episodes details in order to learn
            if learning:
                agent.learn(episode)

            # If the `agent` is a `count_agent` then the agent should update its dynamic betting policy.
            # TODO: This should be added as a function when the agent receives the episode
            if isinstance(agent, count_agent):
                agent.dynamic_bet([agent_hand, dealer_hand])

            # (Optional) only for visualization. If the `agent` is a `human_agent` then the details of the current round
            # are visualized.
            if isinstance(agent, human_agent):
                show(reward, agent_hand, dealer_hand)

            return episode

        # The initial `dealer_hand` and `agent_hand` where not blackjacks. The `agent` proceeds playing.

        # The `agent` plays until either `agent_busted` is True or it decides to stand, which happens iff `action` is
        # set to `s`.
        agent_busted = False
        action = 'h'
        while (not agent_busted) and (not action == 's'):
            # Initialization of the `state` which will be used by the `agent` to decide and update its policies. Note
            # that the agent does not know both the cards of the `dealer` but he has access to only one of them.
            state = [agent_hand, dealer_hand[0]]

            # TODO: Add comments for what happens if `agent` is a `model_based_agent`.
            if isinstance(agent, model_based_agent):
                cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
                concealed_deck = [self.deck.count(value) for value in cards]
                concealed_deck[cards.index(dealer_hand[1])] += 1  # add dealer face down card back in
                action = agent.policy(state, concealed_deck)
            else:
                # The `agent` decides which `action` to take based in its current `state`.
                action = agent.policy(state)
            # Update `episode['actions']` to include the last action that the player decided to take.
            episode['actions'].append(action)
            # If the `action` is 'h' then:
            # (1) the `agent` draws a card from the `self.deck`;
            # (2) `episode['hands']` gets updated; and
            # (3) we check if the agent is busted.
            if action == 'h':
                agent_hand.append(self.draw(agent))  # draw card
                episode['hands'].append(agent_hand.copy())
                agent_busted = self.busted(agent_hand)  # check if busted

            if isinstance(agent, sarsa_agent) and (action == 'h'):  # SARSA is online method => we constantly learn!
                agent.learn(episode) 

        # If the `agent` is busted then round should end (there is not reason for the `dealer` ot play).
        if agent_busted:
            reward = -bet
            if isinstance(agent, human_agent):
                show(reward, agent_hand, dealer_hand)
            if isinstance(agent, count_agent):
                agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
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
            reward = bet
            if isinstance(agent, human_agent):
                show(reward, agent_hand, dealer_hand)
            if isinstance(agent, count_agent):
                agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
            episode['reward'] = reward
            if learning:
                agent.learn(episode)
            return episode

        # At this point we know that both the `agent` and the `dealer` are not busted. Thus we proceed comparing the
        # value of their hands to decide who won and the corresponding reward.
        # If the `agent` won.
        if self.evaluate(agent_hand) > self.evaluate(dealer_hand):
            reward = bet
        # If the `agent` lost.
        elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
            reward = -bet
        # if the game ended as a draw.
        elif self.evaluate(agent_hand) == self.evaluate(dealer_hand):
            reward = 0

        if isinstance(agent, human_agent):
            show(reward, agent_hand, dealer_hand)
        if isinstance(agent, count_agent):
            agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
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
    agent = human_agent()  # interactive agent

    # Play Blackjack
    casino = dealer()
    while True:
        casino.play_round(agent)

        if input('Play again? [y][n]') == 'n':
            break
