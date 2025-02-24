'''
chart_utils.py: Charting utilities for RL experiments.

Functions:
    load_data: Loads data from csv files into lists.
    average_data: Averages data across instances.
    compute_conf_intervals: Confidence interval computation.
    compute_single_conf_interval: Helper function for above.
    _format_title()
    plot: Creates (and opens) a single plot using matplotlib.pyplot
    make_plots: Puts everything in order to create the plot.
    _get_agent_names: Grabs the agent names from parameters.txt.
    _get_agent_colors: Determines the relevant colors/markers for the plot.
    _is_episodic: Determines if the experiment was episodic from parameters.txt.
    _is_disc_reward()
    parse_args: Parse command line arguments.
    main: Loads data from a given path and creates plot.

Author: David Abel (cs.brown.edu/~dabel)
'''

# Python imports.
from __future__ import print_function
import math
import decimal
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import numpy as np
import subprocess
import argparse
from matplotlib.pyplot import MultipleLocator
# import matplotlib
matplotlib.use('Agg')


first_five = [[118, 167, 125], [102, 120, 173], [118, 167, 125], [240, 167, 125], [94, 94, 94]]
# color_ls = [[102, 120, 173], [240, 163, 255], [113, 198, 113],\
#                 [197, 193, 170],[85, 85, 85], [198, 113, 113],\
#                 [142, 56, 142], [125, 158, 192],[184, 221, 255],\
#                 [153, 63, 0], [142, 142, 56], [56, 142, 142]]
color_ls = [[102, 120, 173], [240, 163, 255], [113, 198, 113],\
                [85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]

# Set font.
font = {'size':14}
matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['text.usetex'] = True
fig = matplotlib.pyplot.gcf()

CUSTOM_TITLE = None
X_AXIS_LABEL = None
Y_AXIS_LABEL = None
X_AXIS_START_VAL = 0
X_AXIS_INCREMENT = 1
Y_AXIS_END_VAL = None


def load_data(experiment_dir, experiment_agents):
    '''
    Args:
        experiment_dir (str): Points to the file containing all the data.
        experiment_agents (list): Points to which results files will be plotted.

    Returns:
        result (list): A 3d matrix containing rewards, where the dimensions are [algorithm][instance][episode].
    '''

    result = []
    for alg in experiment_agents:

        # Load the reward for all instances of each agent
        all_reward = open(os.path.join(experiment_dir, str(alg)) + ".csv", "r")
        # all_reward = open(experiment_dir + str(alg) + "__transformed"+".csv", "r")
        all_instances = []

        # Put the reward instances into a list of floats.
        for instance in all_reward.readlines():
            all_episodes_for_instance = [float(r) for r in instance.split(",")[:-1] if len(r) > 0]
            if len(all_episodes_for_instance) > 0:
                all_instances.append(all_episodes_for_instance)

        result.append(all_instances)

    return result


def average_data(data, cumulative=False):
    '''
    Args:
        data (list): a 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt: determines if we should compute the average cumulative reward/cost or just regular.

    Returns:
        (list): a 2D matrix, [algorithm][episode], where the instance rewards have been averaged.
    '''
    num_algorithms = len(data)

    result = [None for i in range(num_algorithms)] # [Alg][avgRewardEpisode], where avg is summed up to episode i if @cumulative=True

    for i, all_instances in enumerate(data):

        # Take the average.
        num_instances = float(len(data[i]))
        all_instances_sum = np.array(np.array(all_instances).sum(axis=0))
        try:
            avged = all_instances_sum / num_instances
        except TypeError:
            raise ValueError("(simple_rl) Plotting Error: an algorithm was run with inconsistent parameters (likely inconsistent number of Episodes/Instances. Try clearing old data).")
        
        if cumulative:
            # If we're summing over episodes.
            temp = []
            total_so_far = 0
            for rew in avged:
                total_so_far += rew
                temp.append(total_so_far)

            avged = temp

        result[i] = avged

    return result


def compute_conf_intervals(data, cumulative=False):
    '''
    Args:
        data (list): A 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt
    '''

    confidence_intervals_each_alg = [] # [alg][conf_inv_for_episode]

    for i, all_instances in enumerate(data):

        num_instances = len(data[i])
        num_episodes = len(data[i][0])

        all_instances_np_arr = np.array(all_instances)
        alg_i_ci = []
        total_so_far = np.zeros(num_instances)
        for j in range(num_episodes):
            # Compute datum for confidence interval.
            episode_j_all_instances = all_instances_np_arr[:, j]

            if cumulative:
                # Cumulative.
                summed_vector = np.add(episode_j_all_instances, total_so_far)
                total_so_far = np.add(episode_j_all_instances, total_so_far)
                episode_j_all_instances = summed_vector

            # Compute the interval and add it to list.
            conf_interv = compute_single_conf_interval(episode_j_all_instances)
            alg_i_ci.append(conf_interv)

        confidence_intervals_each_alg.append(alg_i_ci)

    return confidence_intervals_each_alg

def compute_conf_intervals_task(data, cumulative=False):
    '''
    Args:
        data (list): A 3D matrix, [algorithm][instance][episode]
        cumulative (bool) *opt
    '''

    confidence_intervals_each_alg = [] # [alg][conf_inv_for_episode]

    for i, all_instances in enumerate(data):

        num_instances = len(data[i])
        num_episodes = len(data[i][0])

        all_instances_np_arr = np.array(all_instances)
        alg_i_ci = []
        total_so_far = np.zeros(num_episodes)
        for j in range(num_instances):
            # Compute datum for confidence interval.
            # episode_j_all_instances = all_instances_np_arr[:, j]
            instance_j_all_episodes = all_instances_np_arr[j, :]

            if cumulative:
                # Cumulative.
                summed_vector = np.add(instance_j_all_episodes, total_so_far)
                total_so_far = np.add(instance_j_all_episodes, total_so_far)
                instance_j_all_episodes = summed_vector

            # Compute the interval and add it to list.
            conf_interv = compute_single_conf_interval(instance_j_all_episodes)
            alg_i_ci.append(conf_interv)

        confidence_intervals_each_alg.append(alg_i_ci)

    return confidence_intervals_each_alg

def compute_single_conf_interval(datum):
    '''
    Args:
        datum (list): A vector of data points to compute the confidence interval of.

    Returns:
        (float): Margin of error.
    '''
    std_deviation = np.std(datum)
    std_error = 1.96*(std_deviation / math.sqrt(len(datum)))

    return std_error


def _format_title(plot_title):
    plot_title = plot_title.replace("_", " ")
    plot_title = plot_title.replace("-", " ")
    if len(plot_title.split(" ")) > 1:
        plot_title_final = " ".join([w[0].upper() + w[1:] for w in plot_title.strip().split(" ")])

    return plot_title_final


def plot(results, experiment_dir, agents, plot_file_name="", conf_intervals=[], use_cost=False, cumulative=False,
         episodic=True, open_plot=True, track_disc_reward=False, figure_title=None):
    '''
    Args:
        results (list of lists): each element is itself the reward from an episode for an algorithm.
        experiment_dir (str): path to results.
        agents (list): each element is an agent that was run in the experiment.
        plot_file_name (str)
        conf_intervals (list of floats) [optional]: confidence intervals to display with the chart.
        use_cost (bool) [optional]: If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        cumulative (bool) [optional]: If true, plots are cumulative cost/reward.
        episodic (bool): If true, labels the x-axis "Episode Number". Otherwise, "Step Number".
        open_plot (bool)
        track_disc_reward (bool): If true, plots discounted reward.

    Summary:
        Makes (and opens) a single reward chart plotting all of the data in @data.
    '''

    # Set x-axis labels to be integers.
    from matplotlib.ticker import MaxNLocator
    x_major_locator = MultipleLocator(20)
    y_major_locator = MultipleLocator(1)
    ax = pyplot.figure().gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)

    # Some nice markers and colors for plotting.
    markers = ['o', 's', 'D', '*', '+', 'p', 'x', 'v', '|']

    x_axis_unit = "episode" if episodic else "step"

    # Define colors (assuming color_ls is predefined)
    color_ls = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
                (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)]
    colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]

    # Puts the legend into the best location in the plot and use a tight layout.
    pyplot.rcParams['legend.loc'] = 'best'

    # Negate everything if we're plotting cost.
    if use_cost:
        results = [[-x for x in alg] for alg in results]

    # Dummy function to replace missing _get_agent_colors
    def _get_agent_colors(experiment_dir, agents):
        return {agent: i for i, agent in enumerate(agents)}

    agent_colors = _get_agent_colors(experiment_dir, agents)

    # Make the plot.
    print_prefix = "\nAvg. cumulative reward" if cumulative else "Avg. reward"

    # For each agent.
    for i, agent_name in enumerate(agents):

        # Add figure for this algorithm.
        agent_color_index = i if agent_name not in agent_colors else agent_colors[agent_name]
        series_color = colors[agent_color_index]
        series_marker = markers[agent_color_index]
        y_axis = results[i]
        x_axis = list(range(1, len(y_axis) + 1))
        # Plot Confidence Intervals.
        if conf_intervals != []:
            alg_conf_interv = conf_intervals[i]
            top = np.add(y_axis, alg_conf_interv)
            bot = np.subtract(y_axis, alg_conf_interv)
            pyplot.fill_between(x_axis, top, bot, facecolor=series_color, edgecolor=series_color, alpha=0.25)
        print("\t" + str(agents[i]) + ":", round(y_axis[-1], 5), "(conf_interv:", round(alg_conf_interv[-1], 2), ")")

        marker_every = 3
        pyplot.plot(x_axis, y_axis, color=series_color, marker=series_marker, markersize=4, markevery=marker_every,
                    label=agent_name)

        # Plot legend.
    ax.legend(loc='best', fancybox=True, shadow=True, fontsize=10, ncol=2)

    # Configure plot naming information.
    unit = "Cost" if use_cost else "Reward"
    plot_label = "Cumulative" if cumulative else "Average"
    if "times" in experiment_dir:
        # If it's a time plot.
        unit = "Time"

    disc_ext = "Discounted " if track_disc_reward else ""

    # Set names.
    exp_dir_split_list = experiment_dir.split("/")
    if 'results' in exp_dir_split_list:
        exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
    else:
        exp_name = exp_dir_split_list[0]
    experiment_dir = experiment_dir + "/" if experiment_dir[-1] != "/" else experiment_dir
    plot_file_name = plot_file_name if plot_file_name != "" else experiment_dir + plot_label.lower() + "_" + unit.lower() + ".pdf"
    plot_title = plot_label + " " + disc_ext + unit + ": " + figure_title if figure_title is not None else plot_label + " " + disc_ext + unit + ": " + exp_name
    plot_title = plot_title

    # Axis labels.
    x_axis_label = x_axis_unit[0].upper() + x_axis_unit[1:] + " Number"
    y_axis_label = plot_label + " " + unit
    # Pyplot calls.
    pyplot.xlabel(x_axis_label)
    pyplot.ylabel(y_axis_label)
    pyplot.title(plot_title)
    pyplot.grid(True)

    # Rotate x-axis labels if episode number is large
    # if episodic and len(x_axis) > 20:
    #     ax.set_xticks(x_axis[::len(x_axis) // 10])  # Show a maximum of 20 labels
    #     pyplot.xticks(rotation=0, ha="right")
    if episodic and len(x_axis) > 20:
        # Show a maximum of 10 labels
        max_labels = 10
        step = max(1, len(x_axis) // max_labels)
        ax.set_xticks(np.arange(0, len(x_axis) + 1, step))
        pyplot.xticks(rotation=0, ha="center")
    # Adjust layout to prevent x-axis label from being cut off
    pyplot.tight_layout()

    # Save the plot.
    print(plot_file_name)
    pyplot.savefig(plot_file_name, format='pdf')

    # Clear and close.
    pyplot.cla()
    pyplot.close()

def make_plots(experiment_dir, experiment_agents, plot_file_name="", cumulative=True, use_cost=False, episodic=True, open_plot=True, track_disc_reward=False, figure_title=None):
    '''
    Args:
        experiment_dir (str): path to results.
        experiment_agents (list): agent names (looks for "<agent-name>.csv").
        plot_file_name (str)
        cumulative (bool): If true, plots show cumulative trr
        use_cost (bool): If true, plots are in terms of cost. Otherwise, plots are in terms of reward.
        episodic (bool): If true, labels the x-axis "Episode Number". Otherwise, "Step Number". 
        track_disc_reward (bool): If true, plots discounted reward (changes plot title, too).

    Summary:
        Creates plots for all agents run under the experiment.
        Stores the plot in results/<experiment_name>/<plot_name>.pdf
    '''

    # Load the data.
    data = load_data(experiment_dir, experiment_agents) # [alg][instance][episode]
    # print(np.shape(data))

    # Set sample size for average.
    # data=[i[1:40] for i in data]
    # data=[i[41:80] for i in data]
    # data=[i[1:60] for i in data]
    # print(np.shape(data))

    # Average the data.
    avg_data = average_data(data, cumulative=cumulative)
    # avg_data_task = average_data_task(data, cumulative=cumulative)

    # Compute confidence intervals.
    conf_intervals = compute_conf_intervals(data, cumulative=cumulative)
    # conf_intervals_task = compute_conf_intervals_task(data, cumulative=cumulative)

    # Create plot.
    plot(avg_data, experiment_dir, experiment_agents,
                plot_file_name=plot_file_name,
                conf_intervals=conf_intervals,
                use_cost=use_cost, # False
                cumulative=cumulative, # False
                episodic=episodic, # True
                open_plot=open_plot, # True
                track_disc_reward=track_disc_reward, # False
                figure_title=figure_title)#

def drange(x_min, x_max, x_increment):
    '''
    Args:
        x_min (float)
        x_max (float)
        x_increment (float)

    Returns:
        (generator): Makes a list.

    Notes:
        A range function for generating lists of floats. Based on code from stack overflow user Sam Bruns:
            https://stackoverflow.com/questions/16105485/unsupported-operand-types-for-float-and-decimal
    '''
    x_min = decimal.Decimal(x_min)
    while x_min < x_max:
        yield float(x_min)
        x_min += decimal.Decimal(str(x_increment))

def _get_agent_names(data_dir):
    '''
    Args:
        data_dir (str)

    Returns:
        (list)
    '''
    try:
        params_file = open(os.path.join(data_dir, "parameters.txt"), "r")
    except IOError:
        # No param file.
        return [agent_file.replace(".csv", "") for agent_file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, agent_file)) and ".csv" in agent_file]

    agent_names = []
    agent_flag = False

    for line in params_file.readlines():
        if "Agents" in line:
            agent_flag = True
            continue
        if "Params" in line:
            agent_flag = False
        if agent_flag:
            agent_names.append(line.split(",")[0].strip())

    return agent_names

def _get_agent_colors(data_dir, agents):
    '''
    Args:
        data_dir (str)
        agents (list)

    Returns:
        (list)
    '''
    try:
        params_file = open(os.path.join(data_dir, "parameters.txt"), "r")
    except IOError:
        # No param file.
        return {agent : i for i, agent in enumerate(agents)}

    colors = {}

    # Check if episodes > 1.
    for line in params_file.readlines():
        for agent_name in agents:
            if agent_name == line.strip().split(",")[0]:
                colors[agent_name] = int(line[-2])

    return colors

def _is_episodic(data_dir):
    '''
    Returns:
        (bool) True iff the experiment was episodic.
    '''

    # Open param file for the experiment.
    if not os.path.exists(data_dir + "parameters.txt"):
        print("Warning: no parameters file found for experiment. Assuming non-episodic.")
        return False

    params_file = open(data_dir + "parameters.txt", "r")

    # Check if episodes > 1.
    for line in params_file.readlines():
        if "episodes" in line:
            vals = line.strip().split(":")
            return int(vals[1]) > 1

def _is_disc_reward(data_dir):
    '''
    Returns:
        (bool) True iff the experiment recorded discounted reward.
    '''

    # Open param file for the experiment.
    if not os.path.exists(data_dir + "parameters.txt"):
        print("Warning: no parameters file found for experiment. Assuming non-episodic.")
        return False

    params_file = open(data_dir + "parameters.txt", "r")

    # Check if episodes > 1.
    for line in params_file.readlines():
        if "track_disc_reward" in line:
            vals = line.strip().split(":")
            if "True" == vals[1].strip():
                return True

    return False

def parse_args():
    '''
    Summary:
        Parses two arguments, 'dir' (directory pointer) and 'a' (bool to indicate avg. plot).
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", type = str, help = "Path to relevant csv files of data.")
    parser.add_argument("-a", type = bool, default=False, help = "If true, plots average reward (default is cumulative).")
    return parser.parse_args()

def main():
    '''
    Summary:
        For manual plotting.
    '''
    
    # Parse args.
    args = parse_args()

    # Grab agents.
    data_dir = args.dir
    agent_names = _get_agent_names(data_dir)
    if len(agent_names) == 0:
        raise ValueError("Error: no csv files found.")

    if data_dir[-1] != "/":
        data_dir = data_dir + "/"

    cumulative = not(args.a)
    episodic = _is_episodic(data_dir)
    track_disc_reward = _is_disc_reward(data_dir)

    # Plot.
    make_plots(data_dir, agent_names, cumulative=cumulative, episodic=episodic, track_disc_reward=track_disc_reward)

if __name__ == "__main__":
    main()
