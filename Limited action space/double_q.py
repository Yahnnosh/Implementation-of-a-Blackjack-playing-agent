from agent import agent
import numpy as np
import random
import math

class double_QAgent(agent):

    def __init__(self, alpha=0.01, strategy='random', epsilon=0.5,
                 epsilon_decay=0.99999, temperature=5, ucb_param=2**0.5):
        self.NUMBER_OF_STATES = 363 

        # Initialization: Q1
        self.Q1_S = np.zeros(self.NUMBER_OF_STATES)  # this is Q(State, S)
        self.Q1_H = np.zeros(self.NUMBER_OF_STATES)  # this is Q(State, H)
        self.Q1_H[0], self.Q1_H[1], self.Q1_H[2] = 10, -10, 0   # terminal states
        self.Q1_S[0], self.Q1_S[1], self.Q1_S[2] = 10, -10, 0   # terminal states

        # Initialization: Q2
        self.Q2_S = np.copy(self.Q1_S)
        self.Q2_H = np.copy(self.Q1_H)

        # Initialization: average of Q1, Q2
        self.Q_S = np.copy(self.Q1_S)
        self.Q_H = np.copy(self.Q1_H)
        
        self.gamma = 1  # discount factor
        self.alpha = 0.01   # learning rate
        self.beta = 0.5  # update probability (see double Q algo)

        # Initialization: state visitations
        self.S_visitations = np.zeros(self.NUMBER_OF_STATES)  # N(s, a = stand)
        self.H_visitations = np.zeros(self.NUMBER_OF_STATES)  # N(s, a = hit)

        # Policy params
        assert (strategy == 'random') or (strategy == 'greedy') \
               or (strategy == 'softmax') or (strategy == 'e-greedy') \
               or (strategy == 'ucb')
        self.strategy = strategy  # policy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.temperature = temperature
        self.ucb_param = ucb_param

    def policy(self, hand):
        # state approximation
        state_index = self.state_approx(hand)

        # random policy
        if self.strategy == 'random':
            return random.choice(['h', 's'])

        # greedy policy
        elif self.strategy == 'greedy':
            self.update_q()

            if self.Q_H[state_index] > self.Q_S[state_index]:  # if Q(State, H) > Q(State, S) then we hit
                return 'h'
            elif self.Q_H[state_index] < self.Q_S[state_index]:
                return 's'
            elif self.Q_H[state_index] == self.Q_S[state_index]:
                return random.choice(['h', 's'])

        # softmax policy
        elif self.strategy == 'softmax':
            def softmax(x):
                temperature = 1
                denom = sum([math.exp(temperature * q) for q in x])
                return [math.exp(temperature * q) / denom for q in x]

            self.update_q()
            action = random.choices(population=['h', 's'],
                                    weights=softmax([self.Q_H[state_index], self.Q_S[state_index]]))
            return action[0]

        # epsilon-greedy policy
        elif self.strategy == 'e-greedy':
            self.update_q()

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
            self.update_q()

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

    def update_q(self):
        self.Q_S = (self.Q1_S + self.Q2_S)/2
        self.Q_H = (self.Q1_H + self.Q2_H)/2

    def learn(self, episode):
        actions = episode['actions']
        agent_hands = episode['hands']
        reward = episode['reward']
        dealer_card = episode['dealer'][0][0]

        if not actions: # actions list can be empty if either agent or dealer has a blackjack => nothing to learn here, just return 
            return

        if len(actions) != len(agent_hands):  # then and only then the agent busted
            del agent_hands[-1] # so we remove the last hand of the agent from the episode because it doesn't carry any useful information (it is just lose state)

        if reward > 0:  # if the reward was positive, we won
            final_state_index = 0  # the index of the terminal win state
        elif reward < 0:
            final_state_index = 1  # the index of the terminal lose state
        elif reward == 0:
            final_state_index = 2  # the index of the terminal draw state
       
        while agent_hands:  # while there is something we can learn from

            current_agent_hand = agent_hands.pop(0)  # current hand
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand
            
            current_state_index = self.state_approx([current_agent_hand, dealer_card])  # index of the current state
            next_state_index = self.state_approx([next_agent_hand, dealer_card]) if agent_hands else final_state_index 

            action = actions.pop(0)  # the action which was done in the current state

            # Choose which Q to update (Qa: update, Qb: target)
            if np.random.rand() < self.beta:  # then we update q1
                Qa_H, Qa_S, Qb_H, Qb_S = self.Q1_H, self.Q1_S, self.Q2_H, self.Q2_S
            else:
                Qa_H, Qa_S, Qb_H, Qb_S = self.Q2_H, self.Q2_S, self.Q1_H, self.Q1_S

            # Find max Q value of next state (Qb: target)
            Q_max = max(Qb_H[next_state_index], Qb_S[next_state_index])  # the max Q value for the next state

            # Update Qa based on Qb
            if action == 'h':  # if the action was hit the we update corresponding Q(State, H) function
                Qa_H[current_state_index] += self.alpha * (reward + self.gamma * Q_max - Qa_H[current_state_index])
                self.H_visitations[current_state_index] += 1
            else:
                Qa_S[current_state_index] += self.alpha * (reward + self.gamma * Q_max - Qa_S[current_state_index])
                self.S_visitations[current_state_index] += 1

    def get_Q(self):
        self.update_q()
        return self.Q_H, self.Q_S

    def get_Q_hand(self, hand):
        self.update_q()
        state_index = self.state_approx(hand)
        return self.Q_H[state_index], self.Q_S[state_index]

    def get_visitations(self, hand):
        state_index = self.state_approx(hand)
        return self.H_visitations[state_index], self.S_visitations[state_index]
