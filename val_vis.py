import numpy as np
import pandas as pd
# from matplotlib import pyplot as plt
import seaborn

import sys
import os
import copy
import pprint
from matplotlib import gridspec

import argparse

parser = argparse.ArgumentParser(description='Plot dem results.')
parser.add_argument('--name', default='default')
# parser.add_argument('--keep_losers', default=False)
parser.add_argument('--hide_losers', action='store_true', default=False)
parser.add_argument('--keep_young', action='store_true', default=False)
parser.add_argument('--loser_threshold', default=1)
args = parser.parse_args()

output_dir = "reports/" + args.name

pp = pprint.PrettyPrinter(indent=4)

def mean(l):
    return sum(l) / float(len(l))

networks = {}
for name in sys.stdin:
    network_name = name.strip()
    # print(network_name)
    opt_path = "networks/" + network_name + "/opt.txt"
    loss_path = "networks/" + network_name + "/val_loss.txt"
    # print(os.path.isfile(opt_path))
    # print(os.path.isfile(loss_path))
    try:
        if os.path.isfile(opt_path) and os.path.isfile(loss_path):
            network_data = {}
            with open(opt_path) as opt_file:
                options = {}
                for line in opt_file:
                    k, v = line.split(": ")
                    options[k] = v.strip()
                network_data['options'] = options
                network_data['options']['name'] = network_name

            with open(loss_path) as loss_file:
                losses = []
                for line in loss_file:
                    losses.append(float(line))
                network_data['losses'] = losses

            networks[network_name] = network_data

    except IOError as e:
        pass


if not args.keep_young:
    network_ages = []
    for network_name in networks:
        network = networks[network_name]
        network_ages.append(len(network['losses']))

    mean_network_age = mean(network_ages)

    new_networks = {}
    for network_name in networks:
        network = networks[network_name]
        if len(network['losses']) < (3 * mean_network_age / 4.):
            print("Network is too young. Excluding: " + network_name)
        else:
            new_networks[network_name] = network

    networks = new_networks

if args.hide_losers:
    new_networks = {}
    for network_name in networks:
        network = networks[network_name]
        if network['losses'][-1] > args.loser_threshold:
            print("Network's loss is too high: " + str(network['losses'][-1]) + ". Excluding: " + network_name)
        else:
            new_networks[network_name] = network

    networks = new_networks

same_options = copy.deepcopy(networks[networks.keys()[0]]['options'])
diff_options = []
for network_name in networks:
    network = networks[network_name]
    options = network['options']
    for option in options:
        if option not in diff_options:
            if option not in same_options:
                diff_options.append(option)
            else:
                if options[option] != same_options[option]:
                    diff_options.append(option)
                    same_options.pop(option, None)

print(diff_options)
# print(same_options)

# don't separate them by name
# diff_options.remove("name")

per_option_loss_lists = {}

for option in diff_options:
    option_loss_lists = {}
    for network_name in networks:
        network = networks[network_name]

        option_value = 'none'
        if option in network['options'] and network['options'][option] != '':
            option_value = network['options'][option]

        if option_value not in option_loss_lists:
            option_loss_lists[option_value] = []

        option_loss_lists[option_value].append(network['losses'])

    per_option_loss_lists[option] = option_loss_lists


# per_option_mean_losses = {}
# for option in per_option_loss_lists:
#     per_value_mean_losses = {}
#     for option_value in per_option_loss_lists[option]:
#         loss_lists = per_option_loss_lists[option][option_value]
#
#         last_losses = [losses[-1] for losses in loss_lists]
#         mean_loss = mean(last_losses)
#         per_value_mean_losses[option_value] = mean_loss
#
#     per_option_mean_losses[option] = per_value_mean_losses

per_option_last_losses = {}
for option in per_option_loss_lists:
    per_value_last_losses = {}
    for option_value in per_option_loss_lists[option]:
        loss_lists = per_option_loss_lists[option][option_value]
        per_value_last_losses[option_value] = [losses[-1] for losses in loss_lists]

    per_option_last_losses[option] = per_value_last_losses


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for option in per_option_last_losses:

    if option == 'name' or option == 'import':
        fig = seaborn.plt.figure(figsize=(15,15))
    else:
        fig = seaborn.plt.figure(figsize=(15,10))
    fig.add_subplot()

    df = pd.DataFrame(columns=["option", option, "loss"])
    i = 0
    for option_value in per_option_last_losses[option]:
        for value in per_option_last_losses[option][option_value]:
            df.loc[i] = [option, option_value, value]
            i += 1

    # seaborn.set(font_scale=0.5)
    print(df)
    if option == 'name':
        g = seaborn.barplot(data=df, x=option, y="loss")
    else:
        g = seaborn.boxplot(data=df, x=option, y="loss")
        seaborn.stripplot(data=df, x=option, y="loss", ax=g, color="black")

    if option == 'name' or option == 'import':
        for item in g.get_xticklabels():
            item.set_fontsize(5)


    seaborn.plt.xticks(rotation=90)
    g.set(title=option)
    g.set_yscale('log')

    seaborn.plt.tight_layout()
    seaborn.plt.savefig(output_dir + "/" + option + ".pdf", dpi=300)
    seaborn.plt.close()
