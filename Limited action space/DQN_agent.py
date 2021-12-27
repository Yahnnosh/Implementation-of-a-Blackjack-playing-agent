import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import math
import random
from tqdm import tqdm
import sys

from agent import agent
from dealer import dealer

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Computing on:', device)

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# hyperparameters
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
WEIGHT_DECAY = 0.0001

NUM_LAYERS = 12
k = 13
num_episodes = 800000
start_period = 0
decay_period = num_episodes - 150000 - start_period
learning_rate = 0.0001
# how often to take gradient descent steps
C = 4

graph_interval = 10000

n_in = 4
n_out = 2

# build network
layers_size = {-1: n_in}
factor = (n_out / k / n_in) ** (1 / (NUM_LAYERS - 1))
for layer in range(NUM_LAYERS):
    layers_size[layer] = int(np.rint(k * n_in * factor ** layer))

modules = []
for i in layers_size.keys():
    if i == -1:
        continue
    modules.append(nn.Linear(layers_size[i - 1], layers_size[i]))
    if i < NUM_LAYERS - 1:
        modules.append(nn.BatchNorm1d(layers_size[i]))
        modules.append(nn.ReLU())


class Memory:  # memory buffer
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample(self, batchsize=10):
        return random.sample(self.memory, batchsize)

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return str(self.memory)


class DQN(nn.Module):
    def __init__(self, modules):
        super(DQN, self).__init__()
        for layer, module in enumerate(modules):
            self.add_module("layer_" + str(layer), module)

    def forward(self, x):
        x = x.to(device)
        for layer in self.children():
            x = layer(x)
        return x


class DQNAgent(agent):
    def __init__(self):
        self.model = DQN(modules).to(device)   # neural network with Stanford architecture
        self.memory = Memory(20000)     # memory buffer
        self.gamma = 1
        self.counter = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
        self.training = True

        summary(self.model, (n_in,), device=device)  # visualize model

    def state_approx(self, hand):  # approximates the state as a tuple of 4 elements
        agent_hand = hand[0]
        dealer_hand = hand[1]

        agent_value = self.evaluate(agent_hand)
        dealer_value = self.evaluate([dealer_hand])
        agent_soft = self.soft(agent_hand)
        number_of_cards = len(agent_hand)
        if number_of_cards == 2:
            first_hand = 1  # the first hand
        else:
            first_hand = 0  # not first hand

        # the state is given in the tensor form
        state = torch.tensor([agent_value, dealer_value, agent_soft, first_hand], dtype=torch.float)
        return state

    def policy(self, hand):  # returns next action for the current hand
        state_approx = self.state_approx(hand)
        state_approx = state_approx.view(1, -1)
        self.model.eval()
        Q_stand, Q_hit = self.model(state_approx)[0].detach()

        # greedily return the action
        '''if Q_stand > Q_hit:
            action = 's'
        else:
            action = 'h'

        if self.training:
            prob = random.uniform(0, 1)
            if prob > 0.8:
                return random.choice(['h', 's'])
            else:
                return action
        
        return action'''

        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(- EPS_DECAY)
        if sample > eps_threshold:
            return 'h' if Q_hit > Q_stand else 's'
        else:
            return random.choice(['h', 's'])

    def optimize(self):  # NN updates using a batch of experiences from the memory buffer
        if len(self.memory.memory) < BATCH_SIZE:
            return  # not enough experiences to generate the batch
        self.model.train()  # this tells Pytorch that we are in the training regime
        sample = self.memory.sample(BATCH_SIZE)     # sample a batch of experiences from the buffer
        batch = list(zip(*sample))

        state_batch = torch.stack(batch[0])
        action_batch = batch[1]  # actions of the batch
        reward_batch = torch.tensor(batch[2], device=device)  # rewards of the batch
        next_state_batch = batch[3]

        new_action_batch = []
        action_values = {'s': 0, 'h': 1}
        for action in action_batch:
            new_action_batch.append([action_values[action]])  # here actions are encoded with values
        new_action_batch = torch.tensor(new_action_batch, device=device)  # convert to tensor

        Q_sa = self.model(state_batch)  # compute NN estimates of Q values for the batch for all actions
        Q_sa = Q_sa.gather(1, new_action_batch)  # Q values for all state action pairs from the batch

        mask_next_state_batch = torch.tensor([0 if v is None else 1 for v in next_state_batch], device=device)
        to_nn_next_state_batch = [v if v is not None else torch.tensor([0, 0, 0, 0]) for v in next_state_batch]
        to_nn_next_state_batch = torch.stack(to_nn_next_state_batch).float().to(device)
        V_s = self.model(to_nn_next_state_batch)
        V_s, _ = torch.max(V_s, dim=1)
        V_s = V_s * mask_next_state_batch
        '''observed_sa = reward_batch + self.gamma * V_s'''
        observed_sa = reward_batch + V_s
        observed_sa = observed_sa.view(-1, 1).detach()  # convert to the required size and disable gradient

        loss = F.smooth_l1_loss(Q_sa, observed_sa)  # compute huber loss
        self.optimizer.zero_grad()  # otherwise, gradient would be a combination of previously and newly gradients
        loss.backward()  # calculate all gradients with respect to the weights

        # gradient clipping helps to mitigate exploding gradients issue
        nn.utils.clip_grad_norm(self.model.parameters(), max_norm=2, norm_type=2)

        self.optimizer.step()  # updates weights of NN
        self.model.eval()  # batch normalization layers behave differently for testing and training

    def learn(self, episode):  # NN learns after each episode
        actions = episode['actions']
        agent_hands = episode['hands']
        # reward = episode['reward']
        dealer_card = episode['dealer'][0][0]
        '''
        if self.counter % 1000 == 0:
            print(self.counter)'''

        if not actions:
            return

        if len(actions) != len(agent_hands):
            del agent_hands[-1]  # do not learn from busted hand

        while agent_hands:
            current_agent_hand = agent_hands.pop(0)  # current hand
            next_agent_hand = agent_hands[0] if agent_hands else None  # next hand, none if terminal

            current_state = [current_agent_hand, dealer_card]
            next_state = [next_agent_hand, dealer_card] if agent_hands else None

            current_state = self.state_approx(current_state)
            next_state = self.state_approx(next_state) if agent_hands else None

            if agent_hands:
                reward = 0
            else:
                reward = episode['reward']  # only if next state is the terminal

            action = actions.pop(0)

            experience = (current_state, action, reward, next_state)
            self.memory.push(experience)  # stores the experience in the memory buffer

            if (self.counter % 4) == 0:  # every fourth step
                self.optimize()

            self.counter += 1


if __name__ == '__main__':
    # Initialization
    model = DQNAgent()
    casino = dealer()

    # Select training round
    training_rounds = 1000000

    # Training phase
    loss_per_round = []
    mean_loss_per_round = []
    for t in tqdm(range(training_rounds + 1), leave=False, desc='DQN', file=sys.stdout):
        # every 10% of training rounds show performance
        if (t % (training_rounds / 100) == 0) and (t != 0):
            mean_loss = round(sum(loss_per_round) / (training_rounds / 10), 3)
            print('\nMean loss per round: ', mean_loss, '$')
            mean_loss_per_round.append(mean_loss)
            loss_per_round = []
        episode = casino.play_round(model, bet=1, learning=True)
        reward = episode['reward']
        loss_per_round.append(reward)

    plt.plot(mean_loss_per_round)
    plt.xlabel('rounds')
    plt.ylabel('mean loss per round')
    plt.show()
