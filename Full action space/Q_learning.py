# Q-learning agent for full action space supports the following actions:
# 1) doubling-down
# 2) splitting 
# TODO: 
# 3) insurance 
# 4) surrender 

# do not forget to check allowed actions 

from agent import agent
import numpy as np
import random
import time 

class QAgent(agent):

    def __init__(self, alpha=0.01,  strategy='random', epsilon=0.5,
                 epsilon_decay=0.99999, temperature=5, ucb_param=2**0.5):
        self.NUMBER_OF_STATES = 363  # 3 terminal states + 10 (dealer) * 18 (agent) * 2(soft)

        # Q-values
        self.Stand = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Stand)
        self.Hit = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Hit)
        self.Split = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Split)
        self.Double = np.zeros((self.NUMBER_OF_STATES, 2, 2))  # this is Q(State, Doubling)

        # Q-values of terminal states
        self.Stand[0], self.Stand[1], self.Stand[2] = 1, -1, 0 
        self.Hit[0], self.Hit[1], self.Hit[2] = 1, -1, 0 
        self.Split[0], self.Split[1], self.Split[2] = 1, -1, 0 
        self.Double[0], self.Double[1], self.Double[2] = 1, -1, 0 
        
        self.gamma = 1  # we have a single reward at the end => no need to discount anything
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

    def policy(self, hand, allowed_actions):  # given hand and allowed actions, take a certain action
        old_state_index = self.state_approx(hand)  # this is state index given by old state approximation
        splittable = 1 if 'split' in allowed_actions else 0  # if the agent's hand is splittable or not
        first_hand = int(len(hand[0]) == 2)  # number of cards in an agent hand
        state_index = (old_state_index, first_hand, splittable)

        Q_stand = self.Stand[state_index]
        Q_hit = self.Hit[state_index]
        Q_split = self.Split[state_index]
        Q_double = self.Double[state_index]
       
        # Q-values of all actions for the given state 
        Q_values = {'stand': [Q_stand], 'hit': [Q_hit], 'split': [Q_split], 'double': [Q_double]}  
       
        # Q-values of allowed actions 
        Q_values_allowed = {}
        for key in Q_values:
            if key in allowed_actions:
                Q_values_allowed[key] = Q_values[key]

        # greedy policy
        if self.strategy == 'greedy':
            # Q-values with biggest values
            Q_values_max = {}
            while True:
                key_max = max(Q_values_allowed, key=Q_values_allowed.get)  # the action with biggest Q-value
                Q_values_max[key_max] = Q_values_allowed.pop(key_max)
                if Q_values_allowed:
                    if Q_values_max[key_max] > max(Q_values_allowed.values()): # if Q values of other allowed actions are less
                        break
                else:
                    break

            action = random.choice(list(Q_values_max))  # we choose random action among those with highest Q values
            return action

        # softmax policy
        elif self.strategy == 'softmax':
            def softmax(x):
                temperature = 1
                temp = np.array(x) * temperature
                y = np.exp(temp)
                f_x = y / np.sum(np.exp(temp))
                return f_x

            action = random.choices(population=list(Q_values_allowed),
                                    weights=softmax(list(Q_values_allowed.values())))
            return action[0]

        else:
            raise NotImplementedError

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
            # current state, next state
            current_agent_hand = agent_hands.pop(0)  # current hand
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand

            # state approx: number of cards in hand
            current_number_of_cards = len(current_agent_hand)
            next_number_of_cards = len(next_agent_hand) if next_agent_hand else 2

            # state approx: splittable hand
            current_splittable = 1 if self.split(current_agent_hand) else 0
            next_splittable = 0  # next state is never splittable

            # old state approx
            old_current_state_index = self.state_approx([current_agent_hand, dealer_card])
            old_next_state_index = self.state_approx(
                [next_agent_hand, dealer_card]) if agent_hands else final_state_index

            # new state approximation
            first_hand = lambda x: int(x == 2)
            current_state_index = (old_current_state_index, first_hand(current_number_of_cards), current_splittable)
            next_state_index = (old_next_state_index, first_hand(next_number_of_cards), next_splittable)

            action = actions.pop(0)  # the action which was done in the current state
            Q_max = max(self.Hit[next_state_index], self.Stand[next_state_index])  # the max Q value for the next state

            # and just perform updates of the corresponding Q function
            if action == 'hit':
                self.Hit[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Hit[current_state_index])

            elif action == 'stand':
                self.Stand[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Stand[current_state_index])

            elif action == 'split':
                self.Split[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Split[current_state_index])

            elif action == 'double':
                self.Double[current_state_index] += self.alpha * (
                        reward + self.gamma * Q_max - self.Double[current_state_index])

    def get_Q(self):
        return self.Stand, self.Hit, self.Split, self.Double

    def split(self, hand):
        hand = ['10' if x in ['J', 'Q', 'K'] else x for x in hand] # all face cards are worth ten
        return hand[0] == hand[1]

if __name__ == '__main__':
    agent = QAgent() 
