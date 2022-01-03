from agent import agent
import numpy as np
import random
import math
import pandas as pd

class SARSA_agent(agent):

    def __init__(self, alpha=0.01, strategy='random'):
        self.NUMBER_OF_STATES = 363

        # Initialization: Q-values
        self.Q_S = np.zeros(self.NUMBER_OF_STATES)  # Q(state, stand)
        self.Q_H = np.zeros(self.NUMBER_OF_STATES)  # Q(state, hit)
        self.Q_H[0], self.Q_H[1], self.Q_H[2] = 10, -10, 0  # terminal states
        self.Q_S[0], self.Q_S[1], self.Q_S[2] = 10, -10, 0  # terminal states

        # Initialization: state visitations
        self.S_visitations = np.zeros(self.NUMBER_OF_STATES)  # N(s, stand)
        self.H_visitations = np.zeros(self.NUMBER_OF_STATES)  # N(s, hit)

        self.gamma = 1  # discount factor
        self.alpha = alpha  # learning rate

        # Policy params
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') or (strategy == 'ucb')
        self.strategy = strategy  # policy
        self.epsilon = 0.5
        #self.epsilon_decay = 0.99996  # so that prob(random action) lower than 1% after 100k rounds
        self.epsilon_decay = 0.99999
        self.temperature = 5
        self.ucb_param = 2**0.5

    def policy(self, hand, overwrite=False):
        state_index = self.state_approx(hand) if not overwrite else hand    # current state index

        # random policy
        if self.strategy == 'random':
            return random.choice(['h', 's'])

        # greedy policy
        elif self.strategy == 'greedy':
            if self.Q_H[state_index] > self.Q_S[state_index]:  # if Q(State, H) > Q(State, S) then we hit
                return 'h'
            elif self.Q_H[state_index] < self.Q_S[state_index]:
                return 's'
            elif self.Q_H[state_index] == self.Q_S[state_index]:
                return random.choice(['h', 's'])

        # softmax policy
        elif self.strategy == 'softmax':
            def softmax(x):
                denom = sum([math.exp(self.temperature * q) for q in x])
                return [math.exp(self.temperature * q) / denom for q in x]

            action = random.choices(population=['h', 's'],
                                    weights=softmax([self.Q_H[state_index], self.Q_S[state_index]]))
            return action[0]

        # epsilon-greedy policy
        elif self.strategy == 'e-greedy':
            # act randomly
            if np.random.rand() < self.epsilon:
                action = random.choice(['h', 's'])
            # act greedily
            else:
                if self.Q_H[state_index] > self.Q_S[state_index]:  # if Q(State, H) > Q(State, S) then we hit
                    action = 'h'
                elif self.Q_H[state_index] < self.Q_S[state_index]:
                    action = 's'
                elif self.Q_H[state_index] == self.Q_S[state_index]:
                    action = random.choice(['h', 's'])

            self.epsilon *= self.epsilon_decay
            return action

        # UCB
        elif self.strategy == 'ucb':
            state_visitations = max(1, self.H_visitations[state_index] + self.S_visitations[state_index])
            hit_visitations = max(1, self.H_visitations[state_index])
            stand_visitations = max(1, self.S_visitations[state_index])

            # UCB
            Q_hit = self.Q_H[state_index] + self.ucb_param * (np.log(state_visitations) / hit_visitations) ** 0.5
            Q_stand = self.Q_S[state_index] + self.ucb_param * (np.log(state_visitations) / stand_visitations) ** 0.5

            if Q_hit == Q_stand:
                return random.choice(['h', 's'])
            else:
                return 'h' if Q_hit > Q_stand else 's'

        else:
            raise NotImplementedError

    def activate_greedy(self):
        self.strategy = 'greedy'

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
            # State s_t
            current_agent_hand = agent_hands.pop(0)  # current hand
            current_state_index = self.state_approx([current_agent_hand, dealer_card])  # index of the current state

            # State s_{t+1}
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand
            next_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index

            # Action a_t
            action = actions.pop(0)  # the action which was done in the current state

            # Find Q value of next state
            action_next = self.policy(next_state_index, overwrite=True)
            Q_next = {'h': self.Q_H[next_state_index], 's': self.Q_S[next_state_index]}[action_next]

            # Update Q(s_t, a_t)
            if action == 'h':
                self.Q_H[current_state_index] += self.alpha * (reward + self.gamma * Q_next - self.Q_H[current_state_index])
                self.H_visitations[current_state_index] += 1
            else:
                self.Q_S[current_state_index] += self.alpha * (reward + self.gamma * Q_next - self.Q_S[current_state_index])
                self.S_visitations[current_state_index] += 1

    def get_Q(self):
        return self.Q_H, self.Q_S

    def get_Q_hand(self, hand):
        state_index = self.state_approx(hand)
        return self.Q_H[state_index], self.Q_S[state_index]

    def get_visitations(self, hand):
        state_index = self.state_approx(hand)
        return self.H_visitations[state_index], self.S_visitations[state_index]

    def save_Q(self):
        for action, q in {'hit': self.Q_H, 'stand': self.Q_S}.items():
            name = 'sarsa-' + action
            df = pd.DataFrame(q)
            df.to_csv('Models/' + name + '.csv')

    def load_Q(self, filename_hit, filename_stand):
        Q_H, Q_S = pd.read_csv(filename_hit), pd.read_csv(filename_stand)
        self.Q_H, self.Q_S = list(Q_H.to_numpy()[:, 1]), list(Q_S.to_numpy()[:, 1])
