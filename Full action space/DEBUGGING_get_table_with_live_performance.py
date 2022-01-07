"""
Plots a table for all agent hand / dealer face up card combinations
"""

# Import all agents
from Q_learning_agent import QAgent
from Q_learning_agent_improved import QAgent_improved
from table_policy import table_agent
from SARSA_policy import SARSA_agent

from dealer import dealer
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
from tqdm import trange, tqdm


debugging = False


# TODO: for additional actions
def latexify(actions):
    # colors for actions
    green = '\cellcolor[HTML]{58D68D}'  # stand
    blue = '\cellcolor[HTML]{3498DB}'   # hit

    # header
    print('\n\n'
          '\\begin{table}[]'
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
            color = green if action == 's' else blue
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
    global debugging
    if debugging:
        actions = []
        sublist = []
        visits = []
        visits_sublist = []
        for i, hand in enumerate(hands):
            # make breaks so that table plotting works (convert to matrix form)
            sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double']))
            q_hit, q_stand, q_double, q_split = policy.get_Q_hand(hand)
            hits, stands, doubles, splits = agent.get_visitations(hand)
            visits_sublist.append(
                str(int(hits)) + '(' + str(round(q_hit, 1)) + ')\n'
                + str(int(stands)) + '(' + str(round(q_stand, 1)) + ')\n'
                + str(int(doubles)) + '(' + str(round(q_double, 1)) + ')\n'
                + str(int(splits)) + '(' + str(round(q_split, 1)) + ')')
            if (i + 1) % 10 == 0:
                actions.append(sublist)
                sublist = []
                visits.append(visits_sublist)
                visits_sublist = []
    else:
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
    if debugging:
        blue = '#23427F'    # hit
        green = '#167F45'   # stand
        yellow = '#FFF411'  # split
        violet = '#A911FF'  # double
        color_map = {'hit': blue, 'stand': green, 'split': yellow, 'double': violet}
        colors = [[color_map[value] for value in sublist] for sublist in actions]
        plt.axis('tight')
        plt.axis('off')
        plt.title('Hard hand')
        table = plt.table(cellText=visits,
                  cellColours=colors,
                  rowLabels=rows,
                  colLabels=columns,
                  loc='upper left')
        table.auto_set_font_size(False)
        table.set_fontsize(5)
    else:
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
        table.set_fontsize(6)

    # for LaTeX
    green = '\cellcolor[HTML]{58D68D}'
    blue = '\cellcolor[HTML]{3498DB}'
    orange = '\cellcolor[HTML]{F5B041}'
    counter = 20
    for sublist in actions:
        line = str(counter)
        for action in sublist:
            color = blue
            if action == 'stand':
                color = green
            elif action == 'double':
                color = orange
            line += ' & ' + color + action[0]
        print(line + '\\\\' + '\\hline')
        counter -= 1

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
    global debugging
    if debugging:
        actions = []
        sublist = []
        visits = []
        visits_sublist = []
        for i, hand in enumerate(hands):
            # make breaks so that table plotting works (convert to matrix form)
            sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double']))
            q_hit, q_stand, q_double, q_split = policy.get_Q_hand(hand)
            hits, stands, doubles, splits = agent.get_visitations(hand)
            visits_sublist.append(
                str(int(hits)) + '(' + str(round(q_hit, 1)) + ')\n'
                + str(int(stands)) + '(' + str(round(q_stand, 1)) + ')\n'
                + str(int(doubles)) + '(' + str(round(q_double, 1)) + ')\n'
                + str(int(splits)) + '(' + str(round(q_split, 1)) + ')')
            if (i + 1) % 10 == 0:
                actions.append(sublist)
                sublist = []
                visits.append(visits_sublist)
                visits_sublist = []
    else:
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
    if debugging:
        blue = '#23427F'  # hit
        green = '#167F45'  # stand
        yellow = '#FFF411'  # split
        violet = '#A911FF'  # double
        color_map = {'hit': blue, 'stand': green, 'split': yellow, 'double': violet}
        colors = [[color_map[value] for value in sublist] for sublist in actions]
        plt.axis('tight')
        plt.axis('off')
        plt.title('Soft hand')
        table = plt.table(cellText=visits,
                          cellColours=colors,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='upper left')
        table.auto_set_font_size(False)
        table.set_fontsize(5)
    else:
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
        table.set_fontsize(6)

    # for LaTeX
    green = '\cellcolor[HTML]{58D68D}'
    blue = '\cellcolor[HTML]{3498DB}'
    orange = '\cellcolor[HTML]{F5B041}'
    counter = 20
    for sublist in actions:
        line = str(counter)
        for action in sublist:
            color = blue
            if action == 'stand':
                color = green
            elif action == 'double':
                color = orange
            line += ' & ' + color + action[0]
        print(line + '\\\\' + '\\hline')
        counter -= 1

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
    global debugging
    if debugging:
        actions = []
        sublist = []
        visits = []
        visits_sublist = []
        for i, hand in enumerate(hands):
            # make breaks so that table plotting works (convert to matrix form)
            sublist.append(agent.policy(hand, allowed_actions=['hit', 'stand', 'double', 'split']))
            q_hit, q_stand, q_double, q_split = policy.get_Q_hand(hand)
            hits, stands, doubles, splits = agent.get_visitations(hand)
            visits_sublist.append(
                str(int(hits)) + '(' + str(round(q_hit, 1)) + ')\n'
                + str(int(stands)) + '(' + str(round(q_stand, 1)) + ')\n'
                + str(int(doubles)) + '(' + str(round(q_double, 1)) + ')\n'
                + str(int(splits)) + '(' + str(round(q_split, 1)) + ')')
            if (i + 1) % 10 == 0:
                actions.append(sublist)
                sublist = []
                visits.append(visits_sublist)
                visits_sublist = []
    else:
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
    if debugging:
        blue = '#23427F'
        green = '#167F45'
        yellow = '#FFF411'
        violet = '#A911FF'
        color_map = lambda x: green if x == 'split' else 'white'
        colors = [[color_map(value) for value in sublist] for sublist in actions]
        plt.axis('tight')
        plt.axis('off')
        plt.title('Splittable hand')
        table = plt.table(cellText=visits,
                          cellColours=colors,
                          rowLabels=rows,
                          colLabels=columns,
                          loc='upper left')
        table.auto_set_font_size(False)
        table.set_fontsize(5)
    else:
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
        table.set_fontsize(6)

    # for LaTeX
    yellow = '\cellcolor[HTML]{F4D03F}'
    counter = 20
    for sublist in actions:
        line = str(counter)
        for action in sublist:
            color = yellow if action == 'split' else ''
            a = 's' if action == 'split' else '-'
            line += ' & ' + color + a
        print(line + '\\\\' + '\\hline')
        counter -= 1

if __name__ == '__main__':
    # Pick policy
    policy = SARSA_agent(strategy='softmax')
    policy_name = str(type(policy))[8:].split('.')[0]

    # Pick casino params
    decks = 6
    penetration = 0.8
    infinity = True # if use infinity above is meaningless

    '''for i in range(1000):
        episode = {'hands': [['10', 'Q']], 'dealer': [['3', '4'], ['3', '4', '5'], ['3', '4', '5', 'K']], 'actions': ['stand'], 'reward': 1}
        policy.learn(episode)
        a = policy.Q[policy.state_approx([['10', 'Q'], '3']), 1, 1]
        _, b, _, _ = policy.get_Q_hand(policy.get_Q_hand([['10', 'Q'], '3']))
        print(b)
    exit()'''

    # Training phase
    training_rounds = 1000000
    _RETURN_NONE = (lambda: None).__code__.co_code
    # if the instance has not implemented learn, 'pass' in learn will return None
    if policy.learn.__code__.co_code != _RETURN_NONE:
        print('Starting training')
        casino = dealer(decks=decks, penetration=penetration, infinity=infinity)
        # agent has implemented learn

        # Training phase
        rewards = []
        mean_loss_per_round = []
        # live update
        window = training_rounds // 10  # moving window with size 10% of total training rounds
        with tqdm(total=training_rounds + 1, file=sys.stdout) as pbar:
            for t in range(training_rounds + 1):
                if (t >= window) and (t % window == 0):
                    wins = sum([1 if reward > 0 else 0 for reward in rewards[-window:]])
                    '''mean_win_rate = wins / window
                    pbar.set_description('Mean win rate: ' + str(mean_win_rate))'''

                    mean_loss = round(sum(rewards[-window:]) / window, 3)
                    mean_loss_per_round.append(mean_loss)
                    pbar.set_description('Mean loss per round: ' + str(mean_loss_per_round[-1]))

                episode = casino.play_round(policy, bet=1, learning=True)
                reward = episode['reward']
                rewards.append(reward)

                pbar.update(1)
        print('Finished training for', policy_name)
        fig2 = plt.figure()
        plt.plot(mean_loss_per_round)
    else:
        # agent has not implemented learn
        pass

    policy.activate_greedy()

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
    print(1)    # for debugging (pause)
