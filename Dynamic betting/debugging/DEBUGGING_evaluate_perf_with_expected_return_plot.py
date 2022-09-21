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
from tqdm import trange
from model_based_full_action_space import Model_based_dynamic_betting_policy
from hilo import HiLo

# For full action space :
sys.path.append('../../Full action space')
from table_policy import table_agent  # hard baseline
from Q_learning_agent_improved import QAgent_improved
from sarsa_agent import Sarsa_agent
from dealer import dealer


def simulate(static_policy, dynamic_policy, rounds, ax0, ax1, starting_money):
    """
    Calculates empirical mean win rate, empirical long term profitability
    (i.e. the agent starts with 1000$, the remaining money after the rounds
    is its long term profitability) and the empirical loss per round (mean + std)
    :param static_policy: static betting policy to evaluate
    :param dynamic_policy: dynamic betting policy used to augment static betting policy
    :param rounds: played rounds
    :param ax0: money over time plot
    :param ax1: bet over time plot
    :return: rounded mean win rate, remaining money, mean loss per round, std loss per round
    """

    # params
    casino = dealer()
    bank_account = [starting_money]  # starting money
    n_wins = 0
    game_over = False
    total_rewards = []
    bets = []

    # simulate
    # (tqdm shows progress bar)
    for j in tqdm(range(rounds), leave=False, desc=(get_name(static_policy) + ' + ' + get_name(dynamic_policy)),
                  file=sys.stdout, disable=False):
        # bet: static or dynamic
        if dynamic_policy is None:
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
        episode = casino.play_round(static_policy, bet=bet, learning=False)  # testing independent from training
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

        bets.append(bet)

    # calculate performance params
    mean_win_rate = round(n_wins / rounds, 3)
    long_term_profitability = bank_account[-1]
    mean_loss_per_round = round(sum(total_rewards) / rounds, 3)
    std_loss_per_round = round(np.std(np.array(total_rewards)), 3)

    # plot money evolution + bets
    # padding
    curr_bank_account = bank_account[-1]
    for i in range(rounds + 1 - len(bank_account)):
        bank_account.append(curr_bank_account)  # in case lost all their money

    time_axis = [j for j in range(rounds + 1)]
    # money evolution
    p = ax0.plot(time_axis, bank_account,
             label=(get_name(static_policy) + ' + ' + get_name(dynamic_policy)))
    # bets
    if dynamic_policy is not None:
        ax1.plot(time_axis[1:], bets, color=p[-1].get_color())   # p[-1] is last plotted fct.

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
                if isinstance(policy, HiLo):
                    name += '(' + str(policy.hilo_increment) + ')'
                if isinstance(policy, Model_based_dynamic_betting_policy):
                    if policy.strategy == 'risky':
                        name += '(' + 'infty' ')'
                    else:
                        name += '(' + 'prop' + ')'
                return name
            name += character
        if character == '<':
            writing = True


if __name__ == '__main__':
    # Define casino rules
    min_bet = 1
    max_bet = 100
    increment = 1
    starting_money = 10000

    # Select policies
    # 1) static betting policies
    static_policies = [
        Sarsa_agent()
    ]

    # 2) full policy (static, dynamic)
    full_policies = [
        (static_policies[0], None),
        (static_policies[0], Model_based_dynamic_betting_policy(static_policies[0],
                                                                min_bet=min_bet,
                                                                max_bet=max_bet,
                                                                increment=increment,
                                                                strategy='risky',
                                                                risk=0.56)),
    ]

    # Select rounds
    training_rounds = 1000000
    testing_rounds = 1000000

    # Training phase (static policies)
    print('Starting training')
    _RETURN_NONE = (lambda: None).__code__.co_code
    for static_policy in static_policies:
        # if the instance has not implemented learn, 'pass' in learn will return None
        if static_policy.learn.__code__.co_code != _RETURN_NONE:
            casino = dealer()

            # agent has implemented learn
            for t in trange(training_rounds, desc=get_name(static_policy), file=sys.stdout):
                casino.play_round(static_policy, bet=1, learning=True)  # train agent
            print('Finished training for', get_name(static_policy))
        else:
            # agent has not implemented learn
            pass

    # Testing phase
    print('\nStarting testing')
    fig, (ax0, ax1, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [5, 1, 1]})
    # simulate for each policy
    for tuple in full_policies:
        static = tuple[0]
        dynamic = tuple[1]

        # retrain dynamic policy (to use trained static policy)
        if dynamic is not None:
            dynamic.reset()
            dynamic.record()

        mean_win_rate, long_term_profitability, mean_loss_per_round, std_loss_per_round \
            = simulate(static, dynamic, testing_rounds, ax0, ax1, starting_money)

        print(get_name(static), '+', get_name(dynamic), ':', '\t', mean_win_rate,
              '\t\t', long_term_profitability, '$\t\t', mean_loss_per_round, '$', '(+-', std_loss_per_round, '$)')

    # additional code for plot
    ax0.hlines(starting_money, xmin=0, xmax=testing_rounds, colors='grey', linestyles='dotted')
    ax0.legend(loc='upper right')
    ax0.set_xlabel('rounds')
    ax0.set_ylabel('bank account')

    ax1.set_ylim(min_bet, max_bet)
    ax1.set_xlabel('rounds')
    ax1.set_ylabel('bet')

    for tuple in full_policies:
        static = tuple[0]
        dynamic = tuple[1]

        # retrain dynamic policy (to use trained static policy)
        if dynamic is not None:
            dynamic.show_record(ax3)

    ax3.hlines(0, xmin=0, xmax=testing_rounds, colors='grey', linestyles='dotted')
    ax3.set_xlabel('rounds')
    ax3.set_ylabel('expected return')

    plt.show()
