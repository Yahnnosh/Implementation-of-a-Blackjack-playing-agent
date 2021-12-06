"""
Evaluates the performance of a policy
-------------------------------------
metrics:
1. Empirical mean win rate
2. Empirical long term profitability
3. Empirical loss per round
"""

import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np

sys.path.append('../Limited action space')

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
from model_based import Model_based_dynamic_betting_policy

from dealer import dealer


def simulate(static_policy, dynamic_policy, rounds, plot=False):
    """
    Calculates empirical mean win rate, empirical long term profitability
    (i.e. the agent starts with 1000$, the remaining money after the rounds
    is its long term profitability) and the empirical loss per round (mean + std)
    :param static_policy: static betting policy to evaluate
    :param: dynamic_policy: dynamic betting policy used to augment static betting policy
    :param rounds: played rounds
    :param plot: if True draws the evolution of the bank account on a matplotlib plot
    :return: rounded mean win rate, remaining money, mean loss per round, std loss per round
    """
    # reset card counting agent
    if isinstance(static_policy, count_agent):
        static_policy.reset_counting()

    # params
    casino = dealer()
    bank_account = [1000]   # starting money
    n_wins = 0
    game_over = False
    total_rewards = []

    # simulate
    # (tqdm shows progress bar)
    for j in tqdm(range(rounds), leave=False, desc=str(type(static_policy))[8:].split('.')[0], file=sys.stdout, disable=False):
        # bet: static or dynamic
        if isinstance(static_policy, count_agent):
            bet = static_policy.bet
        elif dynamic_policy is None:
            bet = 1  # static
        else:
            deck = casino.card_counter()
            bet = dynamic_policy.bet(deck)  # dynamic

        # check if enough money for next round
        curr_bank_account = bank_account[-1]
        if curr_bank_account < bet:
            # not enough money to play next round -> stop money change
            game_over = True

        # play one round
        episode = casino.play_round(static_policy, bet=bet, learning=False)    # testing independent from training
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
        plt.plot([j for j in range(rounds + 1)], bank_account, label=str(type(static_policy))[8:].split('.')[0])

    return mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round


if __name__ == '__main__':
    # Select static policies
    static_policies = [
        QAgent(alpha=0.01)
        ]
    policy_names = [str(type(policy))[8:].split('.')[0] for policy in static_policies]
    #TODO: better organized
    # TODO: tuple for each static, dyn combination

    # Select rounds
    training_rounds = 100000
    testing_rounds = 100000

    # Training phase
    print('Starting training')
    _RETURN_NONE = (lambda: None).__code__.co_code
    for i, static_policy in enumerate(static_policies):
        # if the instance has not implemented learn, 'pass' in learn will return None
        if static_policy.learn.__code__.co_code != _RETURN_NONE:
            casino = dealer()
            # agent has implemented learn
            for t in range(training_rounds):
                casino.play_round(static_policy, bet=1, learning=True)  # train agent
            print('Finished training for', policy_names[i])
            # sarsa needs explicit call
            if isinstance(static_policy, sarsa_agent):
                static_policy.set_evaluating()
        else:
            # agent has not implemented learn
            pass

    # Select dynamic policy, params
    min_bet = 1
    max_bet = 100
    increment = 1
    dynamic_policy = Model_based_dynamic_betting_policy(static_policies[0],
                                                        min_bet=min_bet,
                                                        max_bet=max_bet,
                                                        increment=increment)

    # Testing phase
    print('\nStarting testing')
    # for prettier table (aligned)
    max_string_length = max([len(name) for name in policy_names])
    print('-policy name | mean win rate | long term profitability | loss per round-\n')
    # simulate for each policy
    for i, static_policy in enumerate(static_policies):
        mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round \
            = simulate(static_policy, dynamic_policy, testing_rounds, plot=True)
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
