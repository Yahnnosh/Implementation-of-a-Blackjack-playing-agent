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
from card_counting import count_agent       # optimal baseline
from double_q import double_QAgent
from sarsa_agent_old import sarsa_agent
from sarsa_agent import SARSA_agent

from dealer import dealer

import matplotlib.pyplot as plt
from multiprocessing import Process, Manager


def _test(policy, testing_rounds):
    """
    Tests agent over 'training_rounds'
    :param agent: agent to test
    :param testing_rounds: total rounds for testing
    :return: mean win rate over testing_rounds
    """
    # reset card counting agent
    if isinstance(policy, count_agent):
        policy.reset_counting()

    # params
    casino = dealer()
    '''n_wins = 0'''
    total_rewards = []

    # simulate
    for testing_round in range(testing_rounds):
        # play one round
        episode = casino.play_round(policy, bet=1, learning=False)  # testing independent from training
        reward = episode['reward']

        # update params
        '''if reward > 0:
            n_wins += 1'''

        total_rewards.append(reward)

    # calculate performance params
    '''mean_win_rate = round(n_wins / testing_rounds, 3)'''
    mean_loss_per_round = round(sum(total_rewards) / testing_rounds, 3)

    return mean_loss_per_round

def train_with_test(policy, training_rounds, testing_rounds, training_rounds_before_testing, mean_win_rates):
    casino = dealer()
    mean_win_rate = []

    for t in range(training_rounds + 1):
        # test before continue training
        if (t % training_rounds_before_testing == 0) and (t != 0):
            # sarsa needs explicit call
            if isinstance(policy, sarsa_agent):
                policy.set_evaluating()
            mean_win_rate.append(_test(policy, testing_rounds=testing_rounds))
            # sarsa needs explicit call
            if isinstance(policy, sarsa_agent):
                policy.reset_evaluating()

        # train
        casino.play_round(policy, bet=1, learning=True)  # train agent

    print('Finished training for', get_name(policy))
    mean_win_rates[get_name(policy)] = mean_win_rate

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
        double_QAgent(strategy='e-greedy'),
        SARSA_agent(strategy='e-greedy')
    ]

    # Select rounds
    training_rounds = 100000
    testing_rounds = 100000   # the higher the more accurate but will also take longer
    training_rounds_before_testing = 10000  # the higher the smoother the curve but will also take longer

    # Training phase (multiprocessed)
    print('Starting training')

    manager = Manager()
    mean_loss_per_round = manager.dict()
    processes = []
    for policy in policies:
        p = Process(target=train_with_test,
                    args=(policy, training_rounds, testing_rounds, training_rounds_before_testing, mean_loss_per_round))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    for agent in mean_loss_per_round:
        plt.plot([t for t in range(training_rounds_before_testing, training_rounds + 1, training_rounds_before_testing)],
                 mean_loss_per_round[agent], label=agent)
    plt.legend(loc='upper right')
    plt.xlabel('rounds')
    plt.ylabel('mean loss per round')
    plt.hlines(0, xmin=training_rounds_before_testing, xmax=training_rounds, colors='grey', linestyles='dotted')

    plt.show()
