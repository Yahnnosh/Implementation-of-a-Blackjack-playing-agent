import numpy as np
import random
from agent  import agent
#from BlackjackSM import BlackjackSM
#import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from dealer import dealer
import time 


# NN in DQN algorithm takes a state as an input
# and produces Q values for all four actions
# stand / hit / split / doubling 


import numpy as np
import random
#from BlackjackSM import BlackjackSM
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import time

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

# hyperparameters
BATCH_SIZE = 32 # TODO: change to 256 after debugging 
GAMMA = 0.999
# EPS_START = 1
EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 20000
WEIGHT_DECAY = 0.0001

NUM_LAYERS = 4
# k = np.rint(NUM_LAYERS / 2 + 0.5)
k = 13
num_episodes = 800000
# start_period = 50000
start_period = 0
decay_period = num_episodes - 150000 - start_period
learning_rate = 0.0001
# how often to take gradient descent steps
C = 4

graph_interval = 10000

#state_machine = BlackjackSM()
n_in = 4
n_out = 2

# build network
layers_size = {-1: n_in}
factor = (n_out/k/n_in)**(1/(NUM_LAYERS - 1))
for layer in range(NUM_LAYERS):
	layers_size[layer] = int(np.rint(k*n_in * factor**(layer)))


modules = []
for i in layers_size.keys():
	if i == -1: continue
	modules.append(nn.Linear(layers_size[i-1],layers_size[i]))
	if i < NUM_LAYERS - 1:
		modules.append(nn.BatchNorm1d(layers_size[i]))
		modules.append(nn.ReLU())
		# modules.append(nn.Dropout(0.15))



class Memory(): # memory buffer 
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

	def sample(self, batchsize = 10):
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
		for layer in self.children():
			x = layer(x)
		return x

# initialize model
#modell = DQN(modules)
#print(modell)
#time.sleep(10)

#try: 
#	model.load_state_dict(torch.load("models/blackjack_DQN_" + str(NUM_LAYERS) + "-" + str(k) + ".pt"))
#	print("loaded saved model")
#except:
#	print("no saved model")



# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
# memory = Memory(20000)
# model.eval()

# print("check")
# time.sleep(100)



class DQNAgent(agent):

	def __init__(self):
		self.model = DQN(modules) # neural network with Stanford architecture 
		self.memory = Memory(20000) # memory buffer 
		self.gamma = 1
		#self.learning_rate = 1e-2
		#self.training = True
		self.counter = 0
		self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=WEIGHT_DECAY)
		self.training = True 

		# basic NN, next we will perform optimization of its parameters 
		'''
		self.model = (torch.nn.Sequential(
			torch.nn.Linear(4, 8), # first layer takes 5 values: 4 for the state and 1 for the action 
			torch.nn.ReLU(),
			torch.nn.Linear(8, 16),
			torch.nn.ReLU(),
			torch.nn.Linear(16, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 16),
			torch.nn.ReLU(),
			torch.nn.Linear(16, 1),
		))
		'''
		#self.model = (torch.nn.Sequential(
		#	torch.nn.Linear(4, 52), # first layer takes 5 values: 4 for the state and 1 for the action 
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(52, 38),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(27, 20),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(14, 10),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(10, 7),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(7, 5),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(5, 4),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(4, 3),
		#	torch.nn.ReLU(),
		#	torch.nn.Linear(3, 2),
		#))

	def state_approx(self, hand): # approximates the state as a tuple of 4 elements 
		#if action == 's':
		#	action = 0
		#elif action == 'h':
		#	action = 1

		agent_hand = hand[0]
		dealer_hand = hand[1]

		agent_value = self.evaluate(agent_hand)
		dealer_value = self.evaluate([dealer_hand])
		agent_soft = self.soft(agent_hand)
		number_of_cards = len(agent_hand)  
		if number_of_cards == 2:
			first_hand = 1 # the first hand 
		else:
			first_hand = 0 # not first hand 

		# the state is given in the tensor form  
		state = torch.tensor([agent_value, dealer_value, agent_soft, first_hand], dtype=torch.float)
		return state 


	#	def policy(self, hand): # returns next action for the current hand 
	#	state = self.state_approx(hand)
	#	y_pred = self.model(state)
	#	return y_pred
	
	def policy(self, hand): # returns next action for the current hand 
		state_approx = self.state_approx(hand) 
		state_approx = state_approx.view(1,-1)
		self.model.eval()
		Q_stand, Q_hit = self.model(state_approx)[0].detach()
		
		# greedily return the action
		if Q_stand > Q_hit:
			action = 's'
		else:
			action = 'h'
		

		if self.training:
			prob = random.uniform(0,1)
			if prob > 0.8:
				return random.choice(['h', 's'])
			else:
				return action
		
		return action




	def optimize(self): # NN updates using a batch of experiences from the memory buffer 
		if len(self.memory.memory) < BATCH_SIZE:
			return # not enough experiences to generate the batch 	
		self.model.train() # this tells Pytorch that we are in the training regime 
		sample = self.memory.sample(BATCH_SIZE) # sample a batch of experiences from the buffer
		batch = list(zip(*sample))
			
		state_batch = torch.stack(batch[0]) 
		action_batch = batch[1] # actions of the batch
		reward_batch = torch.tensor(batch[2]) # rewards of the batch 
		next_state_batch = batch[3]
	
		new_action_batch = []
		action_values = {'s': 0, 'h': 1}
		for action in action_batch:
			new_action_batch.append([action_values[action]]) # here actions are encoded with values 
		new_action_batch = torch.tensor(new_action_batch) # convert to tensor 

		Q_sa = self.model(state_batch) # compute NN estimates of Q values for the batch for all actions
		Q_sa = Q_sa.gather(1, new_action_batch) # Q values for all state action pairs from the batch 

		mask_next_state_batch = torch.tensor([0 if v is None else 1 for v in next_state_batch])
		to_nn_next_state_batch = [v if v is not None else torch.tensor([0,0,0,0]) for v in next_state_batch]
		to_nn_next_state_batch = torch.stack(to_nn_next_state_batch).float()
		V_s = self.model(to_nn_next_state_batch)
		V_s, _ = torch.max(V_s,dim=1)
		V_s = V_s * mask_next_state_batch
		observed_sa = reward_batch + self.gamma*V_s
		observed_sa = observed_sa.view(-1,1).detach() # convert to the required size and disable gradient

		loss = F.smooth_l1_loss(Q_sa, observed_sa) # compute hubber loss 
		self.optimizer.zero_grad() # otherwise, gradient would be a combination of previously and newly gradients 
		loss.backward() # calculate all gradients with respect to the weights  

		# gradient clipping helps to mitigate exploding gradients issue 
		nn.utils.clip_grad_norm(self.model.parameters(), max_norm=2, norm_type=2) 
		
		self.optimizer.step() # updates weights of NN
		self.model.eval() # batch normalization layers behave differently for testing and training


	def learn(self, episode): # NN learns after each episode 
		actions = episode['actions']
		agent_hands = episode['hands']
		#reward = episode['reward']
		dealer_card = episode['dealer'][0][0]
		if self.counter % 1000 == 0:
			print(self.counter)

		if not actions: 
			return

		if len(actions) != len(agent_hands): 
			del agent_hands[-1] # do not learn from busted hand 

		while agent_hands: 
			current_agent_hand = agent_hands.pop(0) # current hand 
			next_agent_hand = agent_hands[0] if agent_hands else None # next hand, none if terminal 

			current_state = [current_agent_hand, dealer_card] 
			next_state = [next_agent_hand, dealer_card] if agent_hands else None

			current_state = self.state_approx(current_state) 
			next_state = self.state_approx(next_state) if agent_hands else None

			if agent_hands:
				reward = 0
			else:
				reward = episode['reward'] # only if next state is the terminal

			action = actions.pop(0)

			experience = (current_state, action, reward, next_state) 
			self.memory.push(experience) # stores the experince in the memory buffer

			if (self.counter % 4) == 0: # every fourth step
				self.optimize() 

			self.counter += 1


		# TODO: implement the policy method (epsilon greedy)
		# TODO: initialization of the NN? 







			


















			#print('start')
			#print(current_agent_hand)
			#print(next_agent_hand)

			#if agent_hands:
			#	Q_true = self.gamma * self.Q_max(next_state) # this is label if next state is not the terminal 
			#	Q_true = Q_true.detach()
			#else:
			#	Q_true = reward # this is label if the next state is the terminal   

			#action = actions.pop(0)

			#print(Q_true)
			#print(action)
			#time.sleep(10)
			#self.DQN_update(Q_true, current_state, action) # updates NN  



    			         

		        
   
                  
#agent = DQNAgent()
#print(agent.model)
#hand = [['2','3'],'K']
#test = agent.policy(hand)
#test = agent.policy(hand)
#print(test)



# training helper functions

'''
counter = 0
def select_action():
	global counter
	unif_draw = np.random.rand()
	if counter < start_period:
		return LongTensor(np.array([random.choice(state_machine.actions())]))

	eps = EPS_END + max((EPS_START - EPS_END) * (1 - np.exp((counter - start_period - decay_period)/EPS_DECAY)), 0)
	
	scores = model(Variable(FloatTensor(np.array([state_machine.state()])), volatile=True)).data
	mask = ByteTensor(1 - state_machine.mask())
	best_action = (scores.masked_fill_(mask, -16)).max(-1)[1]

	if unif_draw > eps:
		return LongTensor(best_action)
	else:
		actions = state_machine.actions()
		actions.remove(best_action[0])
		return LongTensor(np.array([random.choice(actions)]))

		# actions = state_machine.actions()
		# p = np.abs(scores.numpy()[0])
		# p = (max(np.max(p),1) - p)[actions]
		# p = p / np.sum(p)
		# choice = np.array([np.random.choice(actions, p = p)])
		# return LongTensor(choice)

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return	
	model.train()
	sample = memory.sample(BATCH_SIZE)

	batch = list(zip(*sample))
	state_batch = Variable(torch.cat(batch[0]))
	action_batch = Variable(torch.cat(batch[1]))
	reward_batch = Variable(torch.cat(batch[3]))

	Q_sa = model(state_batch).gather(1, action_batch.view(-1,1)).squeeze()
	
	V_s = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
	not_terminal = ByteTensor(tuple(map(lambda s: s is not None, batch[2])))
	if not_terminal.sum() > 0: 
		model.eval()
		not_terminal_states = Variable(torch.cat([s for s in batch[2] if s is not None]), volatile=True)
		masks = ByteTensor(np.array([1 - state_machine.mask_for(s) for s in not_terminal_states.data.numpy()]))
		V_s[not_terminal] = (model(not_terminal_states).data.masked_fill_(masks, -16)).max(1)[0]
		model.train()
	observed_sa = reward_batch + (V_s * GAMMA)

	loss = F.smooth_l1_loss(Q_sa, observed_sa)

	# pdb.set_trace()

	optimizer.zero_grad()
	loss.backward()
	nn.utils.clip_grad_norm(model.parameters(), max_norm=2, norm_type=2)
	optimizer.step()
	model.eval()

# training
iterations = []
avg_reward_graph = []
avg_reward = 0
avg_score = 0
for episode in range(num_episodes):
	state_machine.new_hand()
	state = FloatTensor([state_machine.state()])

	while True:
		action = select_action()
		state_machine.do(int(action[0]))
		reward = FloatTensor([state_machine.reward()])

		if not state_machine.terminal:
			next_state = FloatTensor([state_machine.state()])
		else:
			next_state = None

		memory.push([state, action, next_state, reward])
		state = next_state

		if counter % C == 0:
			optimize_model()
		counter += 1

		if state_machine.terminal:
			break

	if episode > (num_episodes - 5000):
		avg_score += state_machine.reward() / 5000

	avg_reward += state_machine.reward()
	if episode % graph_interval == 0:
		iterations.append(episode)
		avg_reward_graph.append(avg_reward / graph_interval)
		avg_reward = 0


torch.save(model.state_dict(), "models/blackjack_DQN_" + str(NUM_LAYERS) + "-" + str(k) + ".pt")
print("saved model")

print(avg_score)

import matplotlib.pyplot as plt
plt.title("Average Reward Over Time for a DQN with " + str(NUM_LAYERS) + " Layers and Size Factor " + str(k))
plt.ylabel("return per episode")
plt.xlabel("number of episodes")
plt.plot(iterations, avg_reward_graph)
plt.show()
'''

