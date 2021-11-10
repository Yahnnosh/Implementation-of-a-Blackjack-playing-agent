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

    def check_deck(self, agent):
        if len(self.deck) < (1 - self.PENETRATION) * 52 * self.N_DECKS:  # check if deck over penetration threshold
            self.deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * (4 * self.N_DECKS)
            self.shuffle()

            if isinstance(agent, count_agent):
                agent.reset_counting()  # we also reset counting (in case of the counting agent)

    def draw(self):
        return self.deck.pop()

    def draw_hand(self):
        return [self.draw(), self.draw()]

    def busted(self, hand):
        return sum([self.VALUES[card] for card in hand]) > 21

    def blackjack(self, hand):
        return (len(hand) == 2) and (self.VALUES[hand[0]] + self.VALUES[hand[1]] == 11)

    def evaluate(self, hand):
        val = sum([self.VALUES[card] for card in hand])
        if 'A' in hand:
            return (val + 10) if (val + 10 <= 21) else val  # only 1 ace in hand can ever be used as 11
        else:
            return val

    def soft(self, hand):
        return self.evaluate(hand) - sum([self.VALUES[card] for card in hand]) == 10

    def play_round(self, agent, bet=1):
        """
        Plays one round of Blackjack
        :param agent: deterministic agent - needs to implement policy(state)
        :param bet: betting amount
        :return: full episode of a round (hands, actions, reward)
        """
        # Save episode (for agent learning)
        episode = {'hands': [], 'dealer': [], 'actions': [], 'reward': []}  # dealer always shows first card

        # Check for reshuffle
        self.check_deck(agent)

        # Hand out cards
        agent_hand = self.draw_hand()
        dealer_hand = self.draw_hand()
        episode['hands'].append(agent_hand.copy())
        episode['dealer'].append(dealer_hand.copy())

        # Check for blackjack
        if self.blackjack(dealer_hand):
            reward = 0 if self.blackjack(agent_hand) else -bet
            episode['reward'] = reward
            agent.learn(episode)

            if isinstance(agent, count_agent):
                agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
            # (Optional) only for visualization
            if isinstance(agent, human_agent):
                show(reward, agent_hand, dealer_hand)

            return episode
        if self.blackjack(agent_hand):
            reward = 3 / 2 * bet
            episode['reward'] = reward
            agent.learn(episode)

            if isinstance(agent, count_agent):
                agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round
            # (Optional) only for visualization
            if isinstance(agent, human_agent):
                show(reward, agent_hand, dealer_hand)

            return episode

        # Player turn
        agent_busted = False
        dealer_busted = False
        while True:
            state = [agent_hand, dealer_hand[0]]
            if isinstance(agent, model_based_agent):
                cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
                concealed_deck = [self.deck.count(value) for value in cards]
                concealed_deck[cards.index(dealer_hand[1])] += 1  # add dealer face down card back in
                action = agent.policy(state, concealed_deck)
            else:
                action = agent.policy(state)
            episode['actions'].append(action)

            if action == 's':
                break
            else:
                agent_hand.append(self.deck.pop())  # draw card
                episode['hands'].append(agent_hand.copy())
                agent_busted = self.busted(agent_hand)  # check if busted
                if agent_busted:
                    break

        # Dealer turn
        if not agent_busted:
            while (self.evaluate(dealer_hand) < 17) or \
                    ((self.evaluate(dealer_hand) == 17) and (self.soft(dealer_hand))):  # S17 rule
                dealer_hand.append(self.deck.pop())  # draw card
                episode['dealer'].append(dealer_hand.copy())
                dealer_busted = self.busted(dealer_hand)  # check if busted

        # Payout (win: +bet, lose: -bet, draw: 0)
        reward = 0
        if agent_busted:
            reward = -bet
        elif dealer_busted:
            reward = bet
        else:
            if self.evaluate(agent_hand) > self.evaluate(dealer_hand):
                reward = bet
            elif self.evaluate(agent_hand) < self.evaluate(dealer_hand):
                reward = -bet
            else:
                reward = 0

        # (Optional) only for visualization
        if isinstance(agent, human_agent):
            show(reward, agent_hand, dealer_hand)

        if isinstance(agent, count_agent):
            agent.dynamic_bet([agent_hand, dealer_hand])  # perform counting at the end of the round

        episode['reward'] = reward
        agent.learn(episode)

        return episode

    def card_counter(self):  # TODO: need?
        """
        Returns the probability for the next card for all card values
        :return: [P(2), P(3), P(4), P(5), P(6), P(7), P(8), P(9), P(1)', P(J), P(Q), P(K), P(A)]
        """
        return [self.deck.count(value) / len(self.deck) for value in
                ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']]


if __name__ == '__main__':
    print('Welcome to Blackjack!\n')

    # Policy selection
    # agent = human_agent() # interactive agent
    # agent = random_agent() # random agent
    # agent = table_agent() # fixed-policy (table) agent
    # agent = count_agent() # counting cards (Hi_Lo) agent
    agent = value_iteration()

    # Play Blackjack
    casino = dealer()
    while True:
        # static betting
        casino.play_round(agent)
        # dynamic betting
        '''bet = agent.bet
        casino.play_round(agent, bet)'''

        if input('Play again? [y][n]') == 'n':
            break
