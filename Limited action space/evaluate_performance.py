"""
Evaluates the performance of a policy
-------------------------------------
metrics:
1. Empirical mean win rate
2. Empirical long term profitability
"""

# Import all agents
from random_policy import random_agent      # low baseline
from dealer_policy import dealer_policy     # medium baseline
from table_policy import table_agent        # hard baseline
from card_counting import count_agent       # optimal baseline
from value_iteration import value_iteration
from fast_value_iteration import fast_value_iteration
from Q_learning_agent import QAgent
from sarsa_agent import sarsa_agent
from mc_agent import mc_agent

from dealer import dealer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


def mean_win_rate(policy, rounds):
    """
    Calculate empirical mean win rate (= Exp[win_rate] as rounds -> infty)
    :param policy: policy to evaluate
    :param rounds: played rounds
    :return: rounded win rate
    """
    casino = dealer()
    n_wins = 0
    # tqdm shows progress bar
    for j in tqdm(range(rounds), leave=False, desc=str(type(policy))[8:].split('.')[0], file=sys.stdout):
        episode = casino.play_round(policy, bet=1, learning=False)  # betting amount doesn't matter
        reward = episode['reward']
        if reward > 0:
            n_wins += 1
    return round(n_wins / rounds, 3)

def long_term_profitability(policy, rounds, plot=False):
    """
    Calculates empirical long term profitability, i.e. the agent is given
    1000$, the remaining amount of money after 'rounds' is returned
    :param agent: policy to evaluate
    :param rounds: played rounds
    :param plot: if True draws the evolution of the bank account on a matplotlib plot
    :return: remaining money
    """
    casino = dealer()
    bank_account = [1000]   # starting money
    # tqdm shows progress bar
    for j in tqdm(range(rounds), leave=False, desc=str(type(policy))[8:].split('.')[0], file=sys.stdout):
        bet = policy.bet if isinstance(policy, count_agent) else 1

        curr_bank_account = bank_account[-1]
        if curr_bank_account < bet:
            break

        episode = casino.play_round(policy, bet=bet, learning=False)
        reward = episode['reward']
        bank_account.append(curr_bank_account + reward)

    if plot:
        curr_bank_account = bank_account[-1]
        for i in range(rounds + 1 - len(bank_account)):
            bank_account.append(curr_bank_account) # in case lost all their money
        plt.plot([j for j in range(rounds + 1)], bank_account, label=str(type(policy))[8:].split('.')[0])

    return bank_account[-1]


if __name__ == '__main__':
    # Select policies
    policies = [
        random_agent(),
        dealer_policy(),
        table_agent(),
        count_agent(),
        mc_agent(),
        #sarsa_agent(),
        QAgent(),
        fast_value_iteration()]
    policy_names = [str(type(policy))[8:].split('.')[0] for policy in policies]

    # Select rounds
    testing_rounds = 10000

    # Training phase
    print('Starting training')
    training_rounds = 100000
    _RETURN_NONE = (lambda: None).__code__.co_code
    for i, policy in enumerate(policies):
        # if the instance has not implemented learn, 'pass' in learn will return None
        if policy.learn.__code__.co_code != _RETURN_NONE:
            casino = dealer()
            # agent has implemented learn
            for t in range(training_rounds):
                casino.play_round(policy, bet=1, learning=True) # train agent
            print('Finished training for', policy_names[i])
        else:
            # agent has not implemented learn
            pass

    # Select metric(s)
    print('\nMean win rate:')
    for i, policy in enumerate(policies):
        win_rate = mean_win_rate(policy, testing_rounds)
        print(policy_names[i], ': ', win_rate)

    print('\nLong term profitability:')
    for i, policy in enumerate(policies):
        profitability = long_term_profitability(policy, testing_rounds, plot=True)
        print(policy_names[i], ': ', profitability)
    plt.hlines(1000, xmin=0, xmax=rounds, colors='grey', linestyles='dotted')
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('bank account')
    plt.show()
