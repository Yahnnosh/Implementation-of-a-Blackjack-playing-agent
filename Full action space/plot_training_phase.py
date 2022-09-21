"""
Plots mean win rate over training phase (used for debugging)
"""

"""
Evaluates the performance of a policy
-------------------------------------
metrics:
1. Empirical mean win rate
2. Empirical long term profitability
3. Empirical loss per round
"""

# Import all agents
from sarsa_agent import SARSA_agent

from dealer import dealer

import matplotlib.pyplot as plt
import time
from multiprocessing import Process, Manager


def train(policy, training_rounds, window, transition_length, mean_loss_per_round):
    casino = dealer()
    total_rewards = []
    mean_losses = []
    last_window_timestamp = window - transition_length

    for t in range(training_rounds):
        # play one round
        episode = casino.play_round(policy, bet=1, learning=True)
        reward = episode['reward']

        # update params
        total_rewards.append(reward)

        # moving window
        if (t >= window) and (t - last_window_timestamp == transition_length):
            mean_loss = round(sum(total_rewards[-window:]) / window, 3)
            mean_losses.append(mean_loss)
            last_window_timestamp = t

    # calculate performance params
    mean_loss_per_round[get_name(policy)] = mean_losses
    print('Finished training for', get_name(policy))

def get_name(policy) -> str:
    """
    Returns name of a policy
    :param policy: policy
    :return: name of policy
    """
    if policy is None:
        return 'None'

    name = ''
    writing = False
    for character in str(policy):
        if writing:
            if character == '.':
                if hasattr(policy, 'strategy'):
                    name += '(' + str(policy.strategy) + ')'
                return name
            name += character
        if character == '<':
            writing = True


if __name__ == '__main__':
    t0 = time.time()

    # Select policies
    policies = [
        SARSA_agent(strategy='greedy'),
        SARSA_agent(strategy='softmax'),
        SARSA_agent(strategy='e-greedy'),
        SARSA_agent(strategy='ucb')
    ]

    # Select rounds
    training_rounds = 1000000
    window = training_rounds // 10  # moving window of length 10% of training rounds
    transition_length = 1000  # transition length of next window

    # Training phase (multiprocessed)
    print('Starting training')

    manager = Manager()
    mean_loss_per_round = manager.dict()
    processes = []
    for policy in policies:
        p = Process(target=train,
                    args=(policy, training_rounds, window, transition_length, mean_loss_per_round))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    for agent in mean_loss_per_round:
        print(len(mean_loss_per_round[agent]))
        plt.plot([window + t * transition_length for t in range(len(mean_loss_per_round[agent]))], mean_loss_per_round[agent], label=agent)

    plt.legend(loc='upper right')
    plt.xlabel('rounds')
    plt.ylabel('mean loss per round')
    plt.hlines(0, xmin=window, xmax=training_rounds, colors='grey', linestyles='dotted')

    print('\nTotal time:', time.time() - t0)

    plt.show()
