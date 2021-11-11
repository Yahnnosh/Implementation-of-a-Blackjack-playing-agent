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

from dealer import dealer
import matplotlib.pyplot as plt

def mean_win_rate(policy, rounds):
    """
    Calculate empirical mean win rate (= Exp[win_rate] as rounds -> infty)
    :param policy: policy to evaluate
    :param rounds: played rounds
    :return: rounded win rate
    """
    casino = dealer()
    n_wins = 0
    for i in range(rounds):
        episode = casino.play_round(policy, bet=1)  # betting amount doesn't matter
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
    bank_account = [1000] # starting money
    for i in range(rounds):
        bet = policy.bet if isinstance(policy, count_agent) else 1

        curr_bank_account = bank_account[-1]
        if curr_bank_account < bet:
            break

        episode = casino.play_round(policy, bet=bet)
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
    policies = [random_agent(), dealer_policy(), table_agent(), count_agent(), value_iteration()]
    policy_names = [str(type(policy))[8:].split('.')[0] for policy in policies]

    # Select rounds
    rounds = 100

    # Select metric(s)
    print('Mean win rate:')
    for i, policy in enumerate(policies):
        print(policy_names[i], ': ', mean_win_rate(policy, rounds))

    print('\nLong term profitability:')
    for i, policy in enumerate(policies):
        print(policy_names[i], ': ', long_term_profitability(policy, rounds, plot=True))
    plt.hlines(1000, xmin=0, xmax=rounds, colors='grey', linestyles='dotted')
    plt.legend()
    plt.xlabel('rounds')
    plt.ylabel('bank account')
    plt.show()
