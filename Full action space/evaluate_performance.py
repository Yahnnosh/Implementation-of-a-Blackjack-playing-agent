"""
Evaluates the performance of a policy
-------------------------------------
metrics:
1. Empirical mean win rate
2. Empirical long term profitability
3. Empirical loss per round
"""

# Import all agents
from table_policy import table_agent        # hard baseline
from Q_learning_agent import QAgent

from dealer import dealer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np


def simulate(policy, rounds, plot=False):
    """
    Calculates empirical mean win rate, empirical long term profitability
    (i.e. the agent starts with 1000$, the remaining money after the rounds
    is its long term profitability) and the empirical loss per round (mean + std)
    :param policy: policy to evaluate
    :param rounds: played rounds
    :param plot: if True draws the evolution of the bank account on a matplotlib plot
    :return: rounded mean win rate, remaining money, mean loss per round, std loss per round
    """
    # reset card counting agent
    #if isinstance(policy,count_agent): 
       # policy.reset_counting()

    # params
    casino = dealer()
    bank_account = [1000000]  # starting money
    n_wins = 0
    game_over = False
    total_rewards = []

    # simulate
    # (tqdm shows progress bar)
    for j in tqdm(range(rounds), leave=False, desc=str(type(policy))[8:].split('.')[0], file=sys.stdout, disable=True):
        # bet: static or dynamic
        #bet = policy.bet if isinstance(policy, count_agent) else 1  # TODO: change this for add. dyn. policy
        bet = 1 # the bet is static for now for full action space 

        # check if enough money for next round
        curr_bank_account = bank_account[-1]
        if curr_bank_account < bet:
            # not enough money to play next round -> stop money change
            game_over = True

        # play one round
        episode = casino.play_round(policy, bet=bet, learning=False)    # testing independent from training
        reward = episode['reward']

        # update params
        if game_over:
            last_value = bank_account[-1]
            bank_account.append(last_value)  # money stays at last value (which might be slightly larger than 0)
        else:
            bank_account.append(curr_bank_account + reward)

        if reward > 0:
            n_wins += 1

        total_rewards.append(reward)

    # calculate performance params
    mean_win_rate = round(n_wins / rounds, 3)
    long_term_profitability = bank_account[-1]
    mean_loss_per_round = round(sum(total_rewards) / rounds, 3)
    std_loss_per_round = round(np.std(np.array(total_rewards)), 3)

    # plot money evolution (if demanded)
    if plot:
        curr_bank_account = bank_account[-1]
        for i in range(rounds + 1 - len(bank_account)):
            bank_account.append(curr_bank_account) # in case lost all their money
        plt.plot([j for j in range(rounds + 1)], bank_account, label=str(type(policy))[8:].split('.')[0])

    return mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round


if __name__ == '__main__':
    # Select policies
    policies = [
       # table_agent(),
        QAgent()
        ]
    policy_names = [str(type(policy))[8:].split('.')[0] for policy in policies]

    # Select rounds
    training_rounds = 100
    testing_rounds = 100

    # Training phase
    print('Starting training')
    _RETURN_NONE = (lambda: None).__code__.co_code
    for i, policy in enumerate(policies):
        # if the instance has not implemented learn, 'pass' in learn will return None
        if policy.learn.__code__.co_code != _RETURN_NONE:
            casino = dealer()
            # agent has implemented learn
            for t in range(training_rounds):
                casino.play_round(policy, bet=1, learning=True) # train agent
            print('Finished training for', policy_names[i])
            # sarsa needs explicit call
            if isinstance(policy, sarsa_agent):
                policy.set_evaluating()
        else:
            # agent has not implemented learn
            pass

    # Testing phase
    print('\nStarting testing')
    # for prettier table (aligned)
    max_string_length = max([len(name) for name in policy_names])
    print('-policy name | mean win rate | long term profitability | loss per round-\n')
    # simulate for each policy
    for i, policy in enumerate(policies):
        mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round \
            = simulate(policy, testing_rounds, plot=True)
        # for prettier table (aligned)
        extra_white_space = ''
        for _ in range(max_string_length - len(policy_names[i])):
            extra_white_space += ' '
        print(policy_names[i] + ':' + extra_white_space, '\t', mean_win_rate,
              '\t\t', long_term_profitability, '$\t\t', mean_loss_per_round, '$', '(+-', std_loss_per_round, '$)')
    # additional code for plot
    plt.hlines(1000, xmin=0, xmax=testing_rounds, colors='grey', linestyles='dotted')
    plt.legend(loc='upper right')
    plt.xlabel('rounds')
    plt.ylabel('bank account')
    plt.show()
