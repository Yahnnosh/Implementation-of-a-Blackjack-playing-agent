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
from Q_learning_agent_old import QAgent

from dealer import dealer

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def test(agent, testing_rounds):
    """
    Tests agent over 'training_rounds'
    :param agent: agent to test
    :param testing_rounds: total rounds for testing
    :return: mean loss per round
    """
    # params
    casino = dealer()
    '''n_wins = 0'''
    total_rewards = []

    # simulate
    for testing_round in range(testing_rounds):
        # play one round
        episode = casino.play_round(agent, bet=1, learning=False)  # testing independent from training
        reward = episode['reward']

        # update params
        '''if reward > 0:
            n_wins += 1'''

        total_rewards.append(reward)

    # calculate performance params
    '''mean_win_rate = round(n_wins / testing_rounds, 3)'''
    mean_loss_per_round = round(sum(total_rewards) / testing_rounds, 3)

    return mean_loss_per_round


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
                return name
            name += character
        if character == '<':
            writing = True

if __name__ == '__main__':
    # Select policies
    policies = [
        QAgent()
    ]

    # Select rounds
    training_rounds = 30000000
    testing_rounds = 10000   # the higher the more accurate but will also take longer
    training_rounds_before_testing = 100000   # the higher the smoother the curve but will also take longer

    # Training phase
    print('Starting training')
    for policy in policies:
        casino = dealer()
        loss_per_rounds = []

        for t in tqdm(range(training_rounds), leave=False, desc=get_name(policy), file=sys.stdout, disable=False):
            # test before continue training
            if (t % training_rounds_before_testing == 0) and (t != 0):
                loss_per_rounds.append(test(policy, testing_rounds=testing_rounds))

            # train
            casino.play_round(policy, bet=1, learning=True)  # train agent

        print('Finished training for', get_name(policy))
        plt.plot([i * training_rounds_before_testing for i in range(len(loss_per_rounds))],
                 loss_per_rounds, label=get_name(policy))

    plt.legend(loc='upper right')
    plt.xlabel('rounds')
    plt.ylabel('mean loss per round')

    plt.show()
