"""
Plots a table for all agent hand / dealer face up card combinations
"""

# Import all agents
from Q_learning import QAgent
from Q_learning_agent_old import QAgent
from table_policy import table_agent
from sarsa_agent import SARSA_agent

from dealer import dealer
import matplotlib.pyplot as plt
import math
from tqdm import trange


def latexify(actions, split=False):
    # colors for actions
    green = '\cellcolor[HTML]{58D68D}'  # stand
    blue = '\cellcolor[HTML]{3498DB}'   # hit
    orange = '\cellcolor[HTML]{F5B041}'  # double
    yellow = '\cellcolor[HTML]{F4D03F}'  # split

    # header
    print('\n\n'
          '\\begin{table}[H]'
          '\n\t\centering'
          '\n\t\scalebox{1}{'
          '\n\t\\begin{tabular}{|c|c|c|c|c|c|c|c|c|c|c|}'
          '\n\t\t\hline'
          '\n\t\t& 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10 & A\\\\'
          '\n\t\t\\hline')

    # table content
    counter = 20    # top hand value in table
    for sublist in actions:
        line = '\t\t' + str(counter)
        for action in sublist:
            if split:
                color = yellow if 'split' else ''
                action = 's' if 'split' else '-'
                line += ' & ' + color + action
            else:
                color = {'hit': blue, 'stand': green, 'double': orange, 'split': yellow}[action]
                action = {'hit': 'h', 'stand': 's', 'double': 'd', 'split': 's'}[action]
                line += ' & ' + color + action
        print(line + '\\\\' + '\\hline')
        counter -= 1

    # footer
    print('\t\\end{tabular}'
          '\n\t}'
          '\n\t\\caption{}'
          '\n\\end{table}')

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
        sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double']))
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []

    # Plot table (no 21 as the agent cannot do any action)
    columns = [str(i) for i in range(2, 11)]
    columns.append('A')
    rows = [str(i) for i in range(20, 3, -1)]
    # colors
    blue = '#23427F'    # hit
    green = '#167F45'   # stand
    yellow = '#FFF411'  # split
    violet = '#A911FF'  # double
    color_map = {'hit': blue, 'stand': green, 'split': yellow, 'double': violet}
    colors = [[color_map[value] for value in sublist] for sublist in actions]
    plt.axis('tight')
    plt.axis('off')
    plt.title('Hard hand')
    table = plt.table(cellText=actions,
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # for LaTeX
    latexify(actions)

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
        sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double']))
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []

    # Plot table (no 21 as the agent cannot do any action)
    columns = [str(i) for i in range(2, 11)]
    columns.append('A')
    rows = [str(i) for i in range(20, 12, -1)]
    # colors
    blue = '#23427F'  # hit
    green = '#167F45'  # stand
    yellow = '#FFF411'  # split
    violet = '#A911FF'  # double
    color_map = {'hit': blue, 'stand': green, 'split': yellow, 'double': violet}
    colors = [[color_map[value] for value in sublist] for sublist in actions]
    plt.axis('tight')
    plt.axis('off')
    plt.title('Soft hand')
    table = plt.table(cellText=actions,
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # for LaTeX
    latexify(actions)

def plot_table_split(agent):
    """
    Plots a table for all agent hand / dealer face up card hard combinations
    ------------------------------------------------------------------------
    [4, 2], [4, 3], ..., [4, 11], ..., [20, 2], ..., [20,11]
    :param agent: agent
    :return: None
    """
    # Build possible combinations
    hands = []
    for agent_card in ['A', '10', '9', '8', '7', '6', '5', '4', '3', '2']:  # do in reverse to make table easier to understand
        for dealer_value in range(2, 12):
            # translate value sum back to two cards
            agent_hand = [agent_card] * 2

            # translate dealer ace
            dealer_card = str(dealer_value) if dealer_value != 11 else 'A'

            # complete hand
            hands.append([agent_hand, dealer_card])

    # Get action for each combination
    actions = []
    sublist = []
    for i, hand in enumerate(hands):
        # make breaks so that table plotting works (convert to matrix form)
        sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double', 'split']))
        if (i + 1) % 10 == 0:
            actions.append(sublist)
            sublist = []

    # Plot table (no 21 as the agent cannot do any action)
    columns = [str(i) for i in range(2, 11)]
    columns.append('A')
    #rows = [str(i) for i in range(20, 3, -1)]
    rows = [card + ', ' + card for card in ['A', '10', '9', '8', '7', '6', '5', '4', '3', '2']]
    # colors
    blue = '#23427F'
    green = '#167F45'
    yellow = '#FFF411'
    violet = '#A911FF'
    color_map = lambda x: green if x == 'split' else 'white'
    colors = [[color_map(value) for value in sublist] for sublist in actions]
    plt.axis('tight')
    plt.axis('off')
    plt.title('Splittable hand')
    table = plt.table(cellText=actions,
              cellColours=colors,
              rowLabels=rows,
              colLabels=columns,
              loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # for LaTeX
    latexify(actions, split=True)

if __name__ == '__main__':
    # Pick policy
    policy = SARSA_agent(strategy='ucb')
    policy_name = str(type(policy))[8:].split('.')[0]

    # Pick casino params
    decks = 6
    penetration = 0.8
    infinity = True  # if use infinity above is meaningless

    # Training phase
    training_rounds = 1000000
    _RETURN_NONE = (lambda: None).__code__.co_code
    # if the instance has not implemented learn, 'pass' in learn will return None
    if policy.learn.__code__.co_code != _RETURN_NONE:
        print('Starting training')
        casino = dealer(decks=decks, penetration=penetration, infinity=infinity)
        # agent has implemented learn
        for t in trange(training_rounds, desc=policy_name):
            casino.play_round(policy, bet=1, learning=True)  # train agent
        print('Finished training for', policy_name)
    else:
        # agent has not implemented learn
        pass

    # Plot table hard
    fig = plt.figure()
    fig.suptitle('agent: ' + policy_name)
    plt.subplot(1, 3, 1)
    plot_table_hard(policy)
    plt.subplot(1, 3, 2)
    plot_table_soft(policy)
    plt.subplot(1, 3, 3)
    plot_table_split(policy)
    plt.show()
