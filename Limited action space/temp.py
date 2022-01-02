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
from double_q import double_QAgent
from sarsa_agent import sarsa_agent
from mc_agent import mc_agent
from Q_learning_UCB import QAgent_UCB
from DQN_agent import DQNAgent
from SARSA_policy import SARSA_agent

from dealer import dealer
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import sys

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
    visits = []  # TODO: only for debugging
    visits_sublist = []  # TODO: only for debugging
    for i, hand in enumerate(hands):
        # make breaks so that table plotting works (convert to matrix form)
        sublist.append(agent.policy(hand))
        q_h, q_s = policy.get_Q_hand(hand)   # TODO: only for debugging
        hits, stands = agent.get_visitations(hand)  # TODO: only for debugging
        visits_sublist.append(str(int(hits)) + '(' + str(round(q_h, 1)) + ')\n'
                              + str(int(stands)) + '(' + str(round(q_s, 1)) + ')')  # TODO: only for debugging
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []
            visits.append(visits_sublist)  # TODO: only for debugging
            visits_sublist = []  # TODO: only for debugging

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
    table = plt.table(cellText=visits,  # TODO: only for debugging (otherwise: actions)
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # for LaTeX
    '''green = '\cellcolor[HTML]{58D68D}'
    blue = '\cellcolor[HTML]{3498DB}'
    counter = 20
    for sublist in actions:
        line = str(counter)
        for action in sublist:
            color = green if action == 's' else blue
            line += ' & ' + color + action
        print(line + '\\\\' + '\\hline')
        counter -= 1'''

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
    visits = []   # TODO: only for debugging
    visits_sublist = []  # TODO: only for debugging
    for i, hand in enumerate(hands):
        # make breaks so that table plotting works (convert to matrix form)
        sublist.append(agent.policy(hand))
        q_h, q_s = policy.get_Q_hand(hand)   # TODO: only for debugging
        hits, stands = agent.get_visitations(hand)  # TODO: only for debugging
        visits_sublist.append(str(int(hits)) + '(' + str(round(q_h, 1)) + ')\n'
                              + str(int(stands)) + '(' + str(round(q_s, 1)) + ')')  # TODO: only for debugging
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []
            visits.append(visits_sublist)  # TODO: only for debugging
            visits_sublist = []  # TODO: only for debugging

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
    table = plt.table(cellText=visits,  # TODO: only for debugging (otherwise: actions)
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # for LaTeX
    '''green = '\cellcolor[HTML]{58D68D}'
    blue = '\cellcolor[HTML]{3498DB}'
    counter = 20
    for sublist in actions:
        line = str(counter)
        for action in sublist:
            color = green if action == 's' else blue
            line += ' & ' + color + action
        print(line + '\\\\' + '\\hline')
        counter -= 1'''

if __name__ == '__main__':
    # Pick policy
    #policy = DQNAgent()
    #policy = QAgent_UCB()
    #policy = double_QAgent(alpha=0.01, strategy='e-greedy')
    #policy = QAgent()
    #policy = table_agent()
    #policy = sarsa_agent(strategy='e-greedy')
    #policy = mc_agent(strategy='random')
    policy = SARSA_agent(strategy='softmax')
    policy_name = str(type(policy))[8:].split('.')[0]

    # Training phase
    training_rounds = 1000000
    _RETURN_NONE = (lambda: None).__code__.co_code
    # if the instance has not implemented learn, 'pass' in learn will return None
    if policy.learn.__code__.co_code != _RETURN_NONE:
        print('Starting training')
        casino = dealer()
        # agent has implemented learn
        for t in tqdm(range(training_rounds), leave=False, desc=policy_name,
                      file=sys.stdout, disable=False):
            casino.play_round(policy, bet=1, learning=True)  # train agent
        print('Finished training for', policy_name)
        # sarsa needs explicit call
        if isinstance(policy, sarsa_agent):
            policy.set_evaluating()
    else:
        # agent has not implemented learn
        pass

    #policy.activate_greedy()
    policy.activate_greedy()
    policy.save_Q()

    # Plot table hard
    fig = plt.figure()
    fig.suptitle('agent: ' + policy_name)
    plt.subplot(1, 2, 1)
    plot_table_hard(policy)
    plt.subplot(1, 2, 2)
    plot_table_soft(policy)
    plt.show()
