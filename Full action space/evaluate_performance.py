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
from Q_learning import QAgent
from sarsa_agent import SARSA_agent

from dealer import dealer
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import numpy as np


def simulate(policy, rounds, plot=False, starting_money=1000):
    """
    Calculates empirical mean win rate, empirical long term profitability
    (i.e. the agent starts with 1000$, the remaining money after the rounds
    is its long term profitability) and the empirical loss per round (mean + std)
    :param policy: policy to evaluate
    :param rounds: played rounds
    :param plot: if True draws the evolution of the bank account on a matplotlib plot
    :return: rounded mean win rate, remaining money, mean loss per round, std loss per round
    """

    # params
    casino = dealer()
    bank_account = [starting_money]  # starting money
    n_wins = 0
    game_over = False
    total_rewards = []

    # simulate
    # (tqdm shows progress bar)
    for j in tqdm(range(rounds), leave=False, desc=get_name(policy), file=sys.stdout):
        # bet
        bet = 1

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
        plt.plot([j for j in range(rounds + 1)], bank_account, label=get_name(policy))

    return mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round


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
    # Select policies
    policies = [
        table_agent(),
        QAgent(strategy='random'),
        QAgent(strategy='greedy'),
        QAgent(strategy='softmax'),
        QAgent(strategy='e-greedy'),
        QAgent(strategy='ucb')
        ]

    # Select rounds
    training_rounds = 1000000
    testing_rounds = 1000000

    # Bank account
    money = 10000

    # Training phase
    print('Starting training')
    _RETURN_NONE = (lambda: None).__code__.co_code
    for policy in policies:
        # if the instance has not implemented learn, 'pass' in learn will return None
        if policy.learn.__code__.co_code != _RETURN_NONE:
            casino = dealer()
            # agent has implemented learn
            with tqdm(total=training_rounds + 1, leave=False, desc=get_name(policy), file=sys.stdout) as pbar:
                for t in range(training_rounds):
                    casino.play_round(policy, bet=1, learning=True) # train agent
                    pbar.update(1)
                print('\nFinished training for', get_name(policy))
        else:
            # agent has not implemented learn
            pass

    # Testing phase
    print('\nStarting testing')
    # for prettier table (aligned)
    max_string_length = max([len(get_name(policy)) for policy in policies])
    print('-policy name | mean win rate | long term profitability | loss per round-\n')
    # simulate for each policy
    for policy in policies:
        mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round \
            = simulate(policy, testing_rounds, plot=True, starting_money=money)

        # for prettier table (aligned)
        extra_white_space = ''
        for _ in range(max_string_length - len(get_name(policy))):
            extra_white_space += ' '
        print(get_name(policy) + ':' + extra_white_space, '\t', mean_win_rate,
              '\t\t', long_term_profitability, '$\t\t', mean_loss_per_round, '$', '(+-', std_loss_per_round, '$)')

    # additional code for plot
    plt.hlines(money, xmin=0, xmax=testing_rounds, colors='grey', linestyles='dotted')
    plt.legend(loc='upper right')
    plt.xlabel('rounds')
    plt.ylabel('bank account')
    plt.show()
