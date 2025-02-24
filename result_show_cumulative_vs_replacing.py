#!/usr/bin/env python

###########################################################################################
# Implementation of illustrating results. (Average reward for each task)
# Author for codes: Chu Kun(kun_chu@outlook.com), Abel
# Reference: https://github.com/Kchu/LifelongRL
###########################################################################################

# Python imports.
import os
from simple_rl.utils import chart_utils
from simple_rl.plot_utils import lifelong_plot
from simple_rl.agents.AgentClass import Agent

def _get_MDP_name(data_dir):
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

    MDP_name = []

    for line in params_file.readlines():
        if "lifelong-" in line:
            MDP_name = line.split(" ")[0].strip()
            break

    return MDP_name

def main():
    '''
    Summary:
        For manual plotting.
    '''
    # Parameter
    # data_dir = [r'.\results\Alifelong-four_room_h-11_w-11-q-learning-vs_task\\']
    data_dir = [r'./results/lifelong-rescue_grid_h-19_w-19sarsa_lambda_SR Cumulative vs sarsa_lambda_SR Replacing-q-learning-vs_task/']

    # output_dir = r'.\plots\\'
    output_dir = r'./plots/'
    # Format data dir

    # Grab agents
    
    # Plot.
    for index in range(len(data_dir)): # 就一个
        print('Plotting ' + str(index+1) +'th figure.')
        agent_names = chart_utils._get_agent_names(data_dir[index]) # ['Ideal', 'MaxQInit', 'SR_LLRL', 'Baseline']
        agents = []
        actions = []
        if len(agent_names) == 0:
            raise ValueError("Error: no csv files found.")
        for i in agent_names:
            agent = Agent(i, actions)
            agents.append(agent)

        # Grab experiment settings
        episodic = chart_utils._is_episodic(data_dir[index])
        track_disc_reward = chart_utils._is_disc_reward(data_dir[index])
        mdp_name = _get_MDP_name(data_dir[index]) # 'lifelong-four_room_h-11_w-11'
        lifelong_plot(
            agents,
            data_dir[index],
            output_dir,
            n_tasks=40,
            n_episodes=100,
            confidence=0.95,
            open_plot=True,
            plot_title=True,
            plot_legend=True,
            legend_at_bottom=False,
            episodes_moving_average=False,
            episodes_ma_width=10,
            tasks_moving_average=False,
            tasks_ma_width=10,
            latex_rendering=False,
            figure_title=mdp_name)
if __name__ == "__main__":
    main()