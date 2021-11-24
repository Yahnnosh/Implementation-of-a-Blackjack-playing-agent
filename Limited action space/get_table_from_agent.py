"""
Plots a table for all agent hand / dealer face up card combinations
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
import math

def plot_table_hard(agent):
    """
    Plots a table for all agent hand / dealer face up card hard combinations
    ------------------------------------------------------------------------
    [4, 2], [4, 3], ..., [4, 11], ..., [20, 2], ..., [20,11]
    :param agent: agent
    :return: None
    """
    # Build possible combinations
    hands = []
    for agent_sum in range(20, 3, -1):  # do in reverse to make table easier to understand
        for dealer_value in range(2, 12):
            # translate value sum back to two cards
            agent_hand = [str(math.ceil(agent_sum / 2)), str(math.floor(agent_sum / 2))]

            # translate dealer ace
            dealer_card = str(dealer_value) if dealer_value != 11 else 'A'

            # complete hand
            hands.append([agent_hand, dealer_card])

    # Get action for each combination
    actions = []
    sublist = []
    for i, hand in enumerate(hands):
        # make breaks so that table plotting works (convert to matrix form)
        if isinstance(agent, sarsa_agent):
            sublist.append(agent.policy(hand, evaluating=True))
        else:
            sublist.append(agent.policy(hand))
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []

    # Plot table (no 21 as the agent cannot do any action)
    columns = [str(i) for i in range(2, 11)]
    columns.append('A')
    rows = [str(i) for i in range(20, 3, -1)]
    # colors: hit: blue (#23427F), stand: green (#167F45)
    blue = '#23427F'
    green = '#167F45'
    colors = [[(blue if value == 'h' else green) for value in sublist] for sublist in actions]
    plt.axis('tight')
    plt.axis('off')
    plt.title('Hard hand')
    plt.table(cellText=actions,
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')

def plot_table_soft(agent):
    """
    Plots a table for all agent hand / dealer face up card soft combinations
    ------------------------------------------------------------------------
    [4, 2], [4, 3], ..., [4, 11], ..., [20, 2], ..., [20,11]
    :param agent: agent
    :return: None
    """
    # Build possible combinations
    hands = []
    for agent_card2 in range(9, 1, -1):  # do in reverse to make table easier to understand
        for dealer_value in range(2, 12):
            # translate value sum back to two cards
            agent_hand = ['A', str(agent_card2)]

            # translate dealer ace
            dealer_card = str(dealer_value) if dealer_value != 11 else 'A'

            # complete hand
            hands.append([agent_hand, dealer_card])

    # Get action for each combination
    actions = []
    sublist = []
    for i, hand in enumerate(hands):
        # make breaks so that table plotting works (convert to matrix form)
        if isinstance(agent, sarsa_agent):
            sublist.append(agent.policy(hand, evaluating=True))
        else:
            sublist.append(agent.policy(hand))
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []

    # Plot table (no 21 as the agent cannot do any action)
    columns = [str(i) for i in range(2, 11)]
    columns.append('A')
    rows = [str(i) for i in range(20, 12, -1)]
    # colors: hit: blue (#23427F), stand: green (#167F45)
    blue = '#23427F'
    green = '#167F45'
    colors = [[(blue if value == 'h' else green) for value in sublist] for sublist in actions]
    plt.axis('tight')
    plt.axis('off')
    plt.title('Soft hand')
    plt.table(cellText=actions,
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')

if __name__ == '__main__':
    # Pick policy
    #policy = QAgent()
    #policy = table_agent()
    policy = sarsa_agent()
    #policy = mc_agent()
    policy_name = str(type(policy))[8:].split('.')[0]

    # Training phase
    training_rounds = 100000
    _RETURN_NONE = (lambda: None).__code__.co_code
    # if the instance has not implemented learn, 'pass' in learn will return None
    if policy.learn.__code__.co_code != _RETURN_NONE:
        print('Starting training')
        casino = dealer()
        # agent has implemented learn
        for t in range(training_rounds):
            casino.play_round(policy, bet=1, learning=True)  # train agent
        print('Finished training for', policy_name)
    else:
        # agent has not implemented learn
        pass

    # Plot table hard
    fig = plt.figure()
    fig.suptitle('agent: ' + policy_name)
    plt.subplot(1, 2, 1)
    plot_table_hard(policy)
    plt.subplot(1, 2, 2)
    plot_table_soft(policy)
    plt.show()
