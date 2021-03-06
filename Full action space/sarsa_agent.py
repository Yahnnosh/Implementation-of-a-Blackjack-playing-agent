from agent import agent
import numpy as np
import random
import math
import pandas as pd


# converting csv tables to dictionaries
double_hard_table = pd.read_csv("double_hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard and double is allowed
double_soft_table = pd.read_csv("double_soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft and double is allowed
hard_table = pd.read_csv("hard_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is hard and double is not allowed
soft_table = pd.read_csv("soft_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is soft and double is not allowed
split_table = pd.read_csv("split_table.csv", index_col=0).to_dict()  # the fixed policy if our hand is splittable


class SARSA_agent(agent):

    def __init__(self, alpha=0.005,  strategy='random', epsilon=0.5,
                 epsilon_decay=0.99999, temperature=5, ucb_param=2 ** 0.5):
        self.NUMBER_OF_STATES = 363     # new state approx: (363, splittable, first_hand)
        self.NUMBER_OF_ACTIONS = 4      # hit, stand, split, double

        # Initialization: Q-values
        self.Q = np.zeros((self.NUMBER_OF_STATES, 2, 2, self.NUMBER_OF_ACTIONS))  # Q(s, a)
        self.Q[0], self.Q[1], self.Q[2] = 10, -10, 0  # terminal states

        # Initialization: state visitations
        self.visitations = np.zeros((self.NUMBER_OF_STATES, 2, 2, self.NUMBER_OF_ACTIONS))    # N(s, a)

        self.gamma = 1  # discount factor
        self.alpha = alpha  # learning rate

        # Policy params
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') \
               or (strategy == 'ucb') or (strategy == 'table')
        self.strategy = strategy  # policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.ucb_param = ucb_param

    def action_mapping(self, action):
        return {'hit': 0, 'stand': 1, 'split': 2, 'double': 3}[action]

    def policy(self, hand, allowed_actions, overwrite=False):
        if not overwrite:
            splittable = 1 if self.split(hand[0]) else 0
            first_card = int(len(hand[0]) == 2)
            state_index = (self.state_approx(hand), splittable, first_card)
        else:
            state_index = hand

        # TODO: do not learn insurance - change?
        if 'insurance' in allowed_actions:
            allowed_actions.remove('insurance')

        # random policy
        if self.strategy == 'random':
            return random.choice(allowed_actions)

        # greedy policy
        elif self.strategy == 'greedy':
            q_values = np.array([self.Q[state_index][self.action_mapping(a)] if a in allowed_actions else float('-inf')
                                 for a in ['hit', 'stand', 'split', 'double']])
            action = ['hit', 'stand', 'split', 'double'][np.argmax(q_values)]

            if action not in allowed_actions:  # for unlikely case chose illegal action
                return self.policy(hand, allowed_actions, overwrite=overwrite)
            return action

        # softmax policy
        elif self.strategy == 'softmax':
            def softmax(x):
                denom = sum([math.exp(self.temperature * q) for q in x])
                return [math.exp(self.temperature * q) / denom for q in x]

            q_values = [self.Q[state_index][self.action_mapping(a)] if a in allowed_actions else float('-inf')
                        for a in ['hit', 'stand', 'split', 'double']]
            action = random.choices(population=['hit', 'stand', 'split', 'double'],
                                    weights=softmax(q_values))[0]

            if action not in allowed_actions:  # for unlikely case chose illegal action
                return self.policy(hand, allowed_actions, overwrite=overwrite)
            return action

        # epsilon-greedy policy
        elif self.strategy == 'e-greedy':
            # act randomly
            if np.random.rand() < self.epsilon:
                action = random.choice(['h', 's'])
            # act greedily
            else:
                q_values = np.array([self.Q[state_index][self.action_mapping(a)] if a in allowed_actions else float('-inf')
                                     for a in ['hit', 'stand', 'split', 'double']])
                action = ['hit', 'stand', 'split', 'double'][np.argmax(q_values)]

            self.epsilon *= self.epsilon_decay

            if action not in allowed_actions:  # for unlikely case chose illegal action
                return self.policy(hand, allowed_actions, overwrite=overwrite)
            return action

        # UCB
        elif self.strategy == 'ucb':
            # state action visitations
            state_action_visitations = [max(1, self.visitations[state_index][a]) for a in range(4)]
            state_visitations = sum(state_action_visitations)

            # ucb q values
            q_values = np.array([self.Q[state_index][self.action_mapping(a)] +
                                 self.ucb_param * (np.log(state_visitations) /
                                                   state_action_visitations[self.action_mapping(a)]) ** 0.5
                                 if a in allowed_actions else float('-inf')
                                 for a in ['hit', 'stand', 'split', 'double']])

            action = ['hit', 'stand', 'split', 'double'][np.argmax(q_values)]

            if action not in allowed_actions:  # for unlikely case chose illegal action
                return self.policy(hand, allowed_actions, overwrite=overwrite)
            return action

        elif self.strategy == 'table':
            agent_hand = hand[0]
            dealer_hand = hand[1]

            # translate 10 face values for table use
            if dealer_hand in {"J", "Q", "K"}:
                dealer_hand = "10"

            agent_sum = self.evaluate(agent_hand)  # total card value of the agent

            # check if splittable
            if 'split' in allowed_actions:
                if agent_hand[0] in {"J", "Q", "K"}:
                    agent_hand[0] = "10"
                if split_table[dealer_hand][agent_hand[0]]:  # if splitting recommended
                    return 'split'

            if 'double' in allowed_actions:
                if self.soft(agent_hand):
                    action = double_soft_table[dealer_hand][agent_sum]
                else:
                    action = double_hard_table[dealer_hand][agent_sum]
            else:
                if self.soft(agent_hand):
                    action = soft_table[dealer_hand][agent_sum]
                else:
                    action = hard_table[dealer_hand][agent_sum]

            actions = {
                's': 'stand',
                'h': 'hit',
                'd': 'double'
            }

            action = actions[action]

            return action

        else:
            raise NotImplementedError

    def activate(self, strategy):
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') \
               or (strategy == 'ucb') or (strategy == 'table')
        self.strategy = strategy

    def learn(self, episode):
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions:  # actions list can be empty if either agent or dealer has a blackjack -> nothing to learn
            return

        if len(actions) != len(agent_hands):  # then and only then the agent busted
            del agent_hands[-1]  # busted hand is equal to terminal lose state, do not need it

        if reward > 0:  # if the reward was positive, we won
            final_state_index = 0  # the index of the terminal win state
        elif reward < 0:
            final_state_index = 1  # the index of the terminal lose state
        elif reward == 0:
            final_state_index = 2  # the index of the terminal draw state

        while agent_hands:  # while there is something we can learn from
            # Current state
            current_agent_hand = agent_hands.pop(0)  # current hand
            old_state_index = self.state_approx([current_agent_hand, dealer_card])
            splittable = 1 if self.split(current_agent_hand) else 0
            first_card = int(len(current_agent_hand) == 2)
            current_state_index = (old_state_index, splittable, first_card)

            # Next state
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand
            old_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index
            splittable = 1 if self.split(current_agent_hand) else 0
            first_card = int(len(current_agent_hand) == 2)
            next_state_index = (old_state_index, splittable, first_card)

            # Current action
            action = self.action_mapping(actions.pop(0))  # the action which was done in the current state

            # Q-value of next state
            allowed_actions = ['hit', 'stand']
            if first_card:
                allowed_actions.append('double')
            if splittable:
                allowed_actions.append('split')

            if self.strategy == 'table':
                if next_agent_hand is None:
                    action_next = 0  # action irrelevant for terminal state
                else:
                    action_next = self.action_mapping(self.policy([next_agent_hand, dealer_card], allowed_actions, overwrite=True))
            else:
                action_next = self.action_mapping(self.policy(next_state_index, allowed_actions, overwrite=True))
            Q_next = self.Q[next_state_index][action_next]

            # Update current Q-value
            self.Q[current_state_index][action] += self.alpha * (reward + self.gamma * Q_next - self.Q[current_state_index][action])
            self.visitations[current_state_index][action] += 1

    def split(self, hand):
        if len(hand) != 2:
            return False

        hand = ['10' if x in ['J', 'Q', 'K'] else x for x in hand]  # all face cards are worth ten
        return hand[0] == hand[1]

    def get_Q(self):
        return (self.Q[:, :, :, a] for a in range(4))

    def get_Q_hand(self, hand):
        old_state_index = self.state_approx(hand)
        splittable = 1 if self.split(hand[0]) else 0
        first_card = int(len(hand[0]) == 2)
        state_index = (old_state_index, splittable, first_card)

        return (self.Q[state_index][a] for a in range(4))   # hit, stand, split, double

    def get_visitations(self, hand):
        old_state_index = self.state_approx(hand)
        splittable = 1 if self.split(hand[0]) else 0
        first_card = int(len(hand[0]) == 2)
        state_index = (old_state_index, splittable, first_card)

        return (self.visitations[state_index][a] for a in range(4)) # hit, stand, split, double

    def save_Q(self, name='sars'):
        df = pd.DataFrame(self.Q)
        df.to_csv('Models/' + name + '.csv')

    def load_Q(self, filename):  # TODO: check if works (indices etc.)
        q = pd.read_csv(filename)
        self.Q = list(q.to_numpy()[:, 1])
