#!/usr/bin/env python
'''
Code for running experiments where RL agents interact with an MDP.

Instructions:
    (1) Create an MDP.
    (2) Create agents.
    (3) Set experiment parameters (instances, episodes, steps).
    (4) Call run_agents_on_mdp(agents, mdp) (or the lifelong/markov game equivalents).

    -> Runs all experiments and will open a plot with results when finished.

Author: David Abel (cs.brown.edu/~dabel/)
'''

# Python imports.
from __future__ import print_function
import time
import multiprocessing
import argparse
import os
import math
import sys
import copy
import numpy as np
from collections import defaultdict

# Non-standard imports.
# from simple_rl.planning import ValueIteration
from simple_rl.experiments.ExperimentClass import Experiment
from simple_rl.mdp.markov_game.MarkovGameMDPClass import MarkovGameMDP


# from simple_rl.agents import FixedPolicyAgent
# from simple_rl.plot_utils import lifelong_plot

def print_1D_dict(dic):
    for x in dic:
        print(x, dic[x])


def print_2D_dict(dic):
    for x in dic:
        print(x)
        for y in dic[x]:
            if dic[x][y] > 0:
                print(y, ' ', "%.4f" % dic[x][y], end=' ')
        print("")


def print_3D_dict(dic):
    for x in dic:
        print(x)
        for y in dic[x]:
            print(y)
            for z in dic[x][y]:
                if dic[x][y][z] > 0:
                    print(z, ' ', "%.2f" % dic[x][y][z], end=' ')
        print("")


def instance_reset(agent):
    agent.count_sa = defaultdict(lambda: defaultdict(lambda: 0))
    agent.count_s = defaultdict(lambda: 0)
    agent.episode_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    agent.episode_reward = defaultdict(lambda: 0)
    agent.reward_sa = defaultdict(lambda: defaultdict(lambda: 0))
    agent.task_number = 1
    if agent.q_func is not None:
        agent.q_func = copy.deepcopy(agent.init_q)
        agent.default_q_func = copy.deepcopy(agent.init_q)
    else:
        # agent.q_func_reset()
        agent.reset()


def play_markov_game(agent_ls, markov_game_mdp, instances=10, episodes=100, steps=30, verbose=False, open_plot=True):
    '''
    Args:
        agent_list (list of Agents): See agents/AgentClass.py (and friends).
        markov_game_mdp (MarkovGameMDP): See mdp/markov_games/MarkovGameMDPClass.py.
        instances (int): Number of times to run each agent (for confidence intervals).
        episodes (int): Number of episodes for each learning instance.
        steps (int): Number of times to run each agent (for confidence intervals).
        verbose (bool)
        open_plot (bool): If true opens plot.
    '''

    # Put into dict.
    agent_dict = {}
    for a in agent_ls:
        agent_dict[a.name] = a

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances": instances}  # , "episodes":episodes, "steps":steps}
    experiment = Experiment(agents=agent_dict, mdp=markov_game_mdp, params=exp_params, is_episodic=episodes > 1,
                            is_markov_game=True)

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    # start = time.clock()
    start = time.perf_counter()

    # For each instance of the agent.
    for instance in range(1, instances + 1):
        print("\tInstance " + str(instance) + " of " + str(int(instances)) + ".")

        reward_dict = defaultdict(str)
        action_dict = {}

        for episode in range(1, episodes + 1):
            if verbose:
                sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
                sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
                sys.stdout.flush()

            # Compute initial state/reward.
            state = markov_game_mdp.get_init_state()

            for step in range(steps):

                # Compute each agent's policy.
                for a in agent_dict.values():
                    agent_reward = reward_dict[a.name]
                    agent_action = a.act(state, agent_reward)
                    action_dict[a.name] = agent_action

                # Terminal check.
                if state.is_terminal():
                    experiment.add_experience(agent_dict, state, action_dict, defaultdict(int), state)
                    continue

                # Execute in MDP.
                reward_dict, next_state = markov_game_mdp.execute_agent_action(action_dict)

                # Record the experience.
                experiment.add_experience(agent_dict, state, action_dict, reward_dict, next_state)

                # Update pointer.
                state = next_state

            # A final update.
            for a in agent_dict.values():
                agent_reward = reward_dict[a.name]
                agent_action = a.act(state, agent_reward)
                action_dict[a.name] = agent_action

                # Process that learning instance's info at end of learning.
                experiment.end_of_episode(a.name)

            # Reset the MDP, tell the agent the episode is over.
            markov_game_mdp.reset()

        # A final update.
        for a in agent_dict.values():
            # Reset the agent and track experiment info.
            experiment.end_of_instance(a.name)
            a.reset()

    # Time stuff.
    # print("Experiment took " + str(round(time.clock() - start, 2)) + " seconds.")
    print("Experiment took " + str(round(time.perf_counter() - start, 2)) + " seconds.")

    experiment.make_plots(open_plot=open_plot)

def run_agents_lifelong_test_strategy(agents,
                        mdp_distr,
                        samples=5,
                        episodes=1,
                        steps=100,
                        instances=10,
                        clear_old_results=True,
                        open_plot=True,
                        verbose=False,
                        track_disc_reward=False,
                        reset_at_terminal=False,
                        resample_at_terminal=False,
                        cumulative_plot=True,
                        vs_task=True,
                        alg=None
                        ,env_name=None,
                        obstacle_num=100,
                        is_ablation=False,
                        patient_count=3,
                        pretraining_task_num = 40,
                       val_mdps = None,
                        ):
    '''
    Args:
        agents (list)
        mdp_distr (MDPDistribution)
        samples (int)
        episodes (int)
        steps (int)
        clear_old_results (bool)
        open_plot (bool)
        verbose (bool)
        track_disc_reward (bool): If true records and plots discounted reward, discounted over episodes. So, if
            each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
        reset_at_terminal (bool)
        resample_at_terminal (bool)
        cumulative_plot (bool)

    Summary:
        Runs each agent on the MDP distribution according to the given parameters.
        If @mdp_distr has a non-zero horizon, then gamma is set to 1 and @steps is ignored.
    '''

    # Set number of steps if the horizon is given.
    # if mdp_distr.get_horizon() > 0:
    #     mdp_distr.set_gamma(1.0)
    #     steps = mdp_distr.get_horizon()

    # Experiment (for reproducibility, plotting).
    exp_params = {"samples": samples, "instances": instances, "episodes": episodes, "steps": steps,
                  "gamma": mdp_distr.get_gamma()}
    experiment = Experiment(agents=agents,
                            mdp=mdp_distr,
                            params=exp_params,
                            is_episodic=episodes > 1,
                            is_lifelong=True,
                            clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward,
                            cumulative_plot=cumulative_plot,
                            vs_task=vs_task,
                            alg=alg,
                            detail_name='test_strategy'+ 'obs_num'+str(obstacle_num)+'patient_num'+str(patient_count) if is_ablation else ""
                            )

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    # start = time.clock()
    start = time.perf_counter()

    times = defaultdict(float)
    agent_task_success={}
    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")
        # start = time.clock()
        start = time.perf_counter()
        # data
        experiment.lifelong_save(agent, init=True)  # 为智能体创建一个csv文件。

        # --- SAMPLE NEW MDP FOR EACH INSTANCE ---
        for ins in range(instances):
            # print("    Instance " + str(ins+1) + " of " + str(instances) + ':')
            data = {'returns_per_tasks': [], 'discounted_returns_per_tasks': []}

            task_success = []
            task_count = 0
            for new_task in range(samples):
                task_count += 1
                print("Agent:" + str(agent) + ". Instance: " + str(ins + 1) + ' Total: ' + str(
                    instances) + ". Task: " + str(new_task + 1) + " Total: " + str(samples))
                if task_count == pretraining_task_num:
                    '''
                    perform test:
                    '''
                    # agent.learning = False
                    run_number = 10
                    depth_list = []
                    breadth_list = []
                    for i in range(run_number):
                        # Sample the MDP.
                        # mdp = mdp_distr.sample()
                        mdp = val_mdps.sample()
                        # mdp = mdp_distr.sample()
                        agent.step_number = 0
                        agent.reset()  # reset the agent to let the agent run new task.
                        patient_data_depth = test_strategy_on_mdp(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward,
                                                            reset_at_terminal, resample_at_terminal,None,"depth"
                                                            )
                        mdp.reset()
                        agent.step_number = 0
                        agent.reset()  # reset the agent to let the agent run new task.
                        # patient_data_breadth = test_strategy_on_mdp(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward,
                        #                                     reset_at_terminal, resample_at_terminal,"breadth"
                        #                                     )
                        patient_data_breadth = test_strategy_on_mdp_breadth(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward,
                                                                            reset_at_terminal, resample_at_terminal
                                                                            )
                        agent.step_number = 0
                        agent.reset()  # reset the agent to let the agent run new task.
                        # remove duplicate element, elements type are : [target_x,target_y,found_time]

                        print('depth:',patient_data_depth)
                        print('breadth:',patient_data_breadth)

                        if len(patient_data_depth) > 0:
                            depth_earliest_found_time =patient_data_depth[0][2] if len(patient_data_depth)>0 else -1

                            depth_found_patient_number= len(patient_data_depth)

                            depth_list.append([depth_earliest_found_time,depth_found_patient_number])

                            # depth_list.append([])
                        if len(patient_data_breadth) > 0:
                            breadth_earliest_found_time = patient_data_breadth[0][2] if len(
                                patient_data_breadth) > 0 else -1
                            breadth_found_patient_number = len(patient_data_breadth)
                            breadth_list.append([breadth_earliest_found_time, breadth_found_patient_number])

                    depth_steps = [item[0] for item in depth_list]
                    depth_patients = [item[1] for item in depth_list]

                    breadth_steps = [item[0] for item in breadth_list]
                    breadth_patients = [item[1] for item in breadth_list]

                    # Calculating average steps
                    avg_depth_steps = np.mean(depth_steps)
                    avg_breadth_steps = np.mean(breadth_steps)

                    # Calculating total patients found and patient found rate
                    total_depth_patients = sum(depth_patients)
                    total_breadth_patients = sum(breadth_patients)

                    max_possible_patients = 30  # 3 patients * 10 tests

                    depth_patient_found_rate = total_depth_patients / max_possible_patients
                    breadth_patient_found_rate = total_breadth_patients / max_possible_patients
                    # calculate the avg, and print the agent's depth data and breadth data:
                    # depth_avg = np.mean(np.array(depth_list),axis=0)
                    # breadth_avg = np.mean(np.array(breadth_list),axis=0)
                    # print('depth_first_patient_time:',depth_avg[0],'depth_total_patient_number:',depth_avg[1])
                    # print('breadth_first_patient_time:',breadth_avg[0],'breadth_total_patient_number:',breadth_avg[1])
                    print('depth_list:',depth_list)
                    print('breadth_list:',breadth_list)
                    # failed rate
                    print('avg_depth_steps:',avg_depth_steps,
                          'avg_breadth_steps:',avg_breadth_steps,
                          'depth_patient_found_rate:',depth_patient_found_rate,
                            'breadth_patient_found_rate:',breadth_patient_found_rate
                          )

                    # Assign the filtered lists back to the original variables
                    # patient_data_depth = patient_data_depth_filtered
                    # patient_data_breadth = patient_data_breadth_filtered

                    # print('depth:',patient_data_depth)
                    # print('breadth:',patient_data_breadth)
                    # agent.learning = True
                    exit(0)


                # Sample the MDP.
                mdp = mdp_distr.sample()  # 取出来其中一个任务
                # print(mdp.goal_locs)
                # print(mdp.walls)

                # Run the agent. 这个就是让智能体跑一个任务，agent在线学习, 跑n个episodes,每个episode最多100 steps.
                # returns 是list, discounted_returns 是list. 存储的是每个回合的累积reward 和 discounted reward.
                hit_terminal, total_steps_taken, returns, discounted_returns,success = run_single_agent_on_mdp(agent, mdp,
                                                                                                       episodes, steps,
                                                                                                       experiment,
                                                                                                       verbose,
                                                                                                       track_disc_reward,
                                                                                                       reset_at_terminal,
                                                                                                       resample_at_terminal
                                                                                                       )
                task_success.append(success)

                agent.step_number = 0

                # If we resample at terminal, keep grabbing MDPs until we're done.
                while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                    # resample_at_terminal = false so neven enter this loop
                    mdp = mdp_distr.sample()
                    hit_terminal, steps_taken, returns, discounted_returns = run_single_agent_on_mdp(agent, mdp,
                                                                                                     episodes,
                                                                                                     steps - total_steps_taken,
                                                                                                     experiment,
                                                                                                     verbose,
                                                                                                     track_disc_reward,
                                                                                                     reset_at_terminal,
                                                                                                     resample_at_terminal)
                    total_steps_taken += steps_taken

                data['returns_per_tasks'].append(returns)
                data['discounted_returns_per_tasks'].append(discounted_returns)

                # print_2D_dict(agent.q_func)
                agent.reset()  # reset the agent to let the agent run new task.

            agent_task_success[str(agent)] = task_success

            instance_reset(agent)

            # Track how much time this agent took.
            # end = time.clock()
            end = time.perf_counter()
            times[agent] = round(end - start, 3)
            # print_2D_dict(agent.q_func)
            experiment.lifelong_save(agent, init=False, instance_number=ins, data=data)


    # save task_success with env name:
    name_env = env_name
    # the task_success is a dict, agent name is key, and the value is a list of bool, whether the agent success in each task.
    # so the store style should be column means agent , and row means task.
    # store at results:
    with open('./results/' + name_env + '_task_success.csv', 'w') as f:
        # Write header row
        f.write('task,')
        for agent in agent_task_success:
            f.write(agent + ',')
        f.write('\n')

        # Write task success data
        for i in range(samples):
            f.write(str(i) + ',')
            for agent in agent_task_success:
                f.write(str(agent_task_success[agent][i]) + ',')
            f.write('\n')
        print('task success saved at ./results/' + name_env + '_task_success.csv')
    # Time stuff.
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    # experiment.make_plots(open_plot=open_plot)
    # path = os.path.join(experiment.exp_directory + '/')
    # lifelong_plot(agents, path, output_dir='.\Plots', n_tasks=samples, n_episodes=episodes, confidence=0.95, open_plot=open_plot, plot_title=True, plot_legend=True,
    #             episodes_moving_average=False, episodes_ma_width=10,
    #             tasks_moving_average=False, tasks_ma_width=10, latex_rendering=False)
def run_agents_lifelong(agents,
                        mdp_distr,
                        samples=5,
                        episodes=1,
                        steps=100,
                        instances=10,
                        clear_old_results=True,
                        open_plot=True,
                        verbose=False,
                        track_disc_reward=False,
                        reset_at_terminal=False,
                        resample_at_terminal=False,
                        cumulative_plot=True,
                        vs_task=True,
                        alg=None
                        ,env_name=None,
                        obstacle_num=100,
                        is_ablation=False,
                        patient_count=3,
                        experiment_detail_name=None
                        ):
    '''
    Args:
        agents (list)
        mdp_distr (MDPDistribution)
        samples (int)
        episodes (int)
        steps (int)
        clear_old_results (bool)
        open_plot (bool)
        verbose (bool)
        track_disc_reward (bool): If true records and plots discounted reward, discounted over episodes. So, if
            each episode is 100 steps, then episode 2 will start discounting as though it's step 101.
        reset_at_terminal (bool)
        resample_at_terminal (bool)
        cumulative_plot (bool)

    Summary:
        Runs each agent on the MDP distribution according to the given parameters.
        If @mdp_distr has a non-zero horizon, then gamma is set to 1 and @steps is ignored.
    '''

    # Set number of steps if the horizon is given.
    # if mdp_distr.get_horizon() > 0:
    #     mdp_distr.set_gamma(1.0)
    #     steps = mdp_distr.get_horizon()

    # Experiment (for reproducibility, plotting).
    exp_params = {"samples": samples, "instances": instances, "episodes": episodes, "steps": steps,
                  "gamma": mdp_distr.get_gamma()}
    # detail_content = experiment_detail_name if experiment_detail_name is not None else 'obs_num'+str(obstacle_num)+'patient_num'+str(patient_count) if is_ablation else ""
    detail_content = (
        experiment_detail_name
        if experiment_detail_name is not None
        else f'obs_num{obstacle_num}patient_num{patient_count}'
        if is_ablation
        else ""
    )

    experiment = Experiment(agents=agents,
                            mdp=mdp_distr,
                            params=exp_params,
                            is_episodic=episodes > 1,
                            is_lifelong=True,
                            clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward,
                            cumulative_plot=cumulative_plot,
                            vs_task=vs_task,
                            alg=alg,
                            detail_name=detail_content
                            )

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    # start = time.clock()
    start = time.perf_counter()

    times = defaultdict(float)
    agent_task_success={}
    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")
        # start = time.clock()
        start = time.perf_counter()
        # data
        experiment.lifelong_save(agent, init=True)  # 为智能体创建一个csv文件。

        # --- SAMPLE NEW MDP FOR EACH INSTANCE ---
        for ins in range(instances):
            # print("    Instance " + str(ins+1) + " of " + str(instances) + ':')
            data = {'returns_per_tasks': [], 'discounted_returns_per_tasks': []}

            task_success = []
            for new_task in range(samples):
                print("Agent:" + str(agent) + ". Instance: " + str(ins + 1) + ' Total: ' + str(
                    instances) + ". Task: " + str(new_task + 1) + " Total: " + str(samples))

                # Sample the MDP.
                mdp = mdp_distr.sample()  # 取出来其中一个任务
                # print(mdp.goal_locs)
                # print(mdp.walls)

                # Run the agent. 这个就是让智能体跑一个任务，agent在线学习, 跑n个episodes,每个episode最多100 steps.
                # returns 是list, discounted_returns 是list. 存储的是每个回合的累积reward 和 discounted reward.
                hit_terminal, total_steps_taken, returns, discounted_returns,success = run_single_agent_on_mdp(agent, mdp,
                                                                                                       episodes, steps,
                                                                                                       experiment,
                                                                                                       verbose,
                                                                                                       track_disc_reward,
                                                                                                       reset_at_terminal,
                                                                                                       resample_at_terminal
                                                                                                       )
                task_success.append(success)

                agent.step_number = 0

                # If we resample at terminal, keep grabbing MDPs until we're done.
                while resample_at_terminal and hit_terminal and total_steps_taken < steps:
                    # resample_at_terminal = false so neven enter this loop
                    mdp = mdp_distr.sample()
                    hit_terminal, steps_taken, returns, discounted_returns = run_single_agent_on_mdp(agent, mdp,
                                                                                                     episodes,
                                                                                                     steps - total_steps_taken,
                                                                                                     experiment,
                                                                                                     verbose,
                                                                                                     track_disc_reward,
                                                                                                     reset_at_terminal,
                                                                                                     resample_at_terminal)
                    total_steps_taken += steps_taken

                data['returns_per_tasks'].append(returns)
                data['discounted_returns_per_tasks'].append(discounted_returns)

                # print_2D_dict(agent.q_func)
                agent.reset()  # reset the agent to let the agent run new task.

            agent_task_success[str(agent)] = task_success

            instance_reset(agent)

            # Track how much time this agent took.
            # end = time.clock()
            end = time.perf_counter()
            times[agent] = round(end - start, 3)
            # print_2D_dict(agent.q_func)
            experiment.lifelong_save(agent, init=False, instance_number=ins, data=data)


    # save task_success with env name:
    name_env = env_name
    # the task_success is a dict, agent name is key, and the value is a list of bool, whether the agent success in each task.
    # so the store style should be column means agent , and row means task.
    # store at results:
    with open('./results/' + name_env + '_task_success.csv', 'w') as f:
        # Write header row
        f.write('task,')
        for agent in agent_task_success:
            f.write(agent + ',')
        f.write('\n')

        # Write task success data
        for i in range(samples):
            f.write(str(i) + ',')
            for agent in agent_task_success:
                f.write(str(agent_task_success[agent][i]) + ',')
            f.write('\n')
        print('task success saved at ./results/' + name_env + '_task_success.csv')
    # Time stuff.
    print("\n--- TIMES ---")
    for agent in times.keys():
        print(str(agent) + " agent took " + str(round(times[agent], 2)) + " seconds.")
    print("-------------\n")

    # experiment.make_plots(open_plot=open_plot)
    # path = os.path.join(experiment.exp_directory + '/')
    # lifelong_plot(agents, path, output_dir='.\Plots', n_tasks=samples, n_episodes=episodes, confidence=0.95, open_plot=open_plot, plot_title=True, plot_legend=True,
    #             episodes_moving_average=False, episodes_ma_width=10, 
    #             tasks_moving_average=False, tasks_ma_width=10, latex_rendering=False)


def run_agents_on_mdp(agents,
                      mdp,
                      instances=5,
                      episodes=100,
                      steps=200,
                      clear_old_results=True,
                      rew_step_count=1,
                      track_disc_reward=False,
                      open_plot=True,
                      verbose=False,
                      reset_at_terminal=False,  # this is true in main setting.
                      cumulative_plot=True):
    '''
    Args:
        agents (list of Agents): See agents/AgentClass.py (and friends).
        mdp (MDP): See mdp/MDPClass.py for the abstract class. Specific MDPs in tasks/*.
        instances (int): Number of times to run each agent (for confidence intervals).
        episodes (int): Number of episodes for each learning instance.
        steps (int): Number of steps per episode.
        clear_old_results (bool): If true, removes all results files in the relevant results dir.
        rew_step_count (int): Number of steps before recording reward.
        track_disc_reward (bool): If true, track (and plot) discounted reward.
        open_plot (bool): If true opens the plot at the end.
        verbose (bool): If true, prints status bars per episode/instance.
        reset_at_terminal (bool): If true sends the agent to the start state after terminal.
        cumulative_plot (bool): If true makes a cumulative plot, otherwise plots avg. reward per timestep.

    Summary:
        Runs each agent on the given mdp according to the given parameters.
        Stores results in results/<agent_name>.csv and automatically
        generates a plot and opens it.
    '''

    # Experiment (for reproducibility, plotting).
    exp_params = {"instances": instances, "episodes": episodes, "steps": steps, "gamma": mdp.get_gamma()}
    experiment = Experiment(
        agents=agents,
                            mdp=mdp,
                            params=exp_params,
                            is_episodic=episodes > 1,
                            clear_old_results=clear_old_results,
                            track_disc_reward=track_disc_reward,
                            count_r_per_n_timestep=rew_step_count,
                            cumulative_plot=cumulative_plot
                            )

    # Record how long each agent spends learning.
    print("Running experiment: \n" + str(experiment))
    time_dict = defaultdict(float)

    # Learn.
    for agent in agents:
        print(str(agent) + " is learning.")

        # start = time.clock()
        start = time.perf_counter()

        # For each instance.
        for instance in range(1, instances + 1):
            print("  Instance " + str(instance) + " of " + str(instances) + ".")
            sys.stdout.flush()
            run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment, verbose, track_disc_reward,
                                    reset_at_terminal=reset_at_terminal)

            # Reset the agent.
            agent.reset()
            mdp.end_of_instance()

        # Track how much time this agent took.
        # end = time.clock()
        end = time.perf_counter()
        time_dict[agent] = round(end - start, 3)
        print()

    # Time stuff.
    print("\n--- TIMES ---")
    for agent in time_dict.keys():
        print(str(agent) + " agent took " + str(round(time_dict[agent], 2)) + " seconds.")
    print("-------------\n")

    experiment.make_plots(open_plot=open_plot)


def run_single_agent_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False,task_success_list=None):
    '''
    Summary:
        Main loop of a single MDP experiment.  跑一个task.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, float: cumulative discounted reward)
    '''
    if reset_at_terminal and resample_at_terminal:
        raise ValueError(
            "(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")
    success = False
    value = 0
    gamma = mdp.get_gamma()
    return_per_episode = [0] * episodes
    dicounted_return_per_episode = [0] * episodes

    # For each episode.
    for episode in range(1, episodes + 1):
        '''
        把200,000分成 200 episodes x 100 steps原因是:每隔一个episode就把agent就更新anneal.
        '''

        if verbose:
            # Print episode numbers out nicely.
            sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
            sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
            sys.stdout.flush()

        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0
        # episode_start_time = time.clock()
        episode_start_time = time.perf_counter()

        # Extra printing if verbose.
        if verbose:
            print()
            sys.stdout.flush()
            prog_bar_len = _make_step_progress_bar()

        for step in range(1, steps + 1):
            # print(step)
            if verbose and int(prog_bar_len * float(step) / steps) > int(prog_bar_len * float(step - 1) / steps):
                _increment_bar()

            # step time
            # step_start = time.clock()
            step_start = time.perf_counter()

            # # Compute the agent's policy.
            action = agent.act(state, reward)
            # Terminal check.
            if state.is_terminal():
                if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    # experiment.add_experience(agent, state, action, 0, state, time_taken = time.clock()-step_start)
                    experiment.add_experience(agent, state, action, 0, state,
                                              time_taken=time.perf_counter() - step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)
            agent.episode_count[episode][state][action] += 1
            agent.episode_reward[episode] += reward
            return_per_episode[episode - 1] += reward

            # Track value.
            value += reward * gamma ** step
            dicounted_return_per_episode[episode - 1] += reward * gamma ** step

            # Record the experience.
            if experiment is not None: # 不会进去的分支。
                reward_to_track = mdp.get_gamma() ** (
                            step + 1 + episode * steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)
                # experiment.add_experience(agent, state, action, reward_to_track, next_state, time_taken=time.clock() - step_start)
                experiment.add_experience(agent, state, action, reward_to_track, next_state,
                                          time_taken=time.perf_counter() - step_start)

            if next_state.is_terminal():  #
                success = True
                if reset_at_terminal:  # true
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    agent.step_number = 0
                    mdp.reset()
                elif resample_at_terminal and step < steps:
                    agent.step_number = 0
                    mdp.reset()
                    return True, steps, return_per_episode, dicounted_return_per_episode

            # Update pointer.
            state = next_state

        # A final update.
        action = agent.act(state, reward)

        # Process experiment info at end of episode.
        if experiment is not None:
            experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        if verbose:
            print("\n")

    # Get the trajectory which has the maxmium reward.
    max_r = float("-inf")
    best_e = 0

    # when task finished, calculate collected s,a for reward shaping.
    for episode in range(1, episodes + 1):
        if agent.episode_reward[episode] > max_r:
            max_r = agent.episode_reward[episode]  # 这个episode(是这100帧)采样到的最大 accumulated reward.
            best_e = episode  # 记录这个回合的序号。

    for x in agent.episode_count[best_e]:  # 只记录表现最好的那个回合的s,a
        for y in agent.episode_count[best_e][x]:
            agent.count_sa[x][y] += agent.episode_count[best_e][x][y]
            agent.count_s[x] += agent.episode_count[best_e][x][y]
            # agent.count +=agent.episode_count[best_e][x][y]
    agent.episode_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # 记录完之后重置
    agent.episode_reward = defaultdict(lambda: 0)

    # Process that learning instance's info at end of learning.
    if experiment is not None:
        experiment.end_of_instance(agent)

    return False, steps, return_per_episode, dicounted_return_per_episode,success

def test_strategy_on_mdp(agent, mdp, episodes, steps, experiment=None, verbose=False, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False,task_success_list=None,strategy="depth"):
    '''
    Summary:
        Main loop of a single MDP experiment.  跑一个task.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, float: cumulative discounted reward)
    '''
    if reset_at_terminal and resample_at_terminal:
        raise ValueError(
            "(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")
    success = False
    value = 0
    gamma = mdp.get_gamma()
    return_per_episode = [0] * episodes
    dicounted_return_per_episode = [0] * episodes


    reward = 0
    # episode_start_time = time.clock()
    episode_start_time = time.perf_counter()
    patient_found = []
    # For each episode.
    state = mdp.get_init_state()
    total_step = 0
    for episode in range(1, episodes + 1):
        '''
        把200,000分成 200 episodes x 100 steps原因是:每隔一个episode就把agent就更新anneal.
        '''

        # if verbose:
        #     # Print episode numbers out nicely.
        #     sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
        #     sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
        #     sys.stdout.flush()

        # Compute initial state/reward.




        # Extra printing if verbose.
        if verbose:
            print()
            sys.stdout.flush()
            prog_bar_len = _make_step_progress_bar()

        for step in range(1, steps + 1):

            # print(step)
            if verbose and int(prog_bar_len * float(step) / steps) > int(prog_bar_len * float(step - 1) / steps):
                _increment_bar()

            # step time
            # step_start = time.clock()
            step_start = time.perf_counter()

            # # Compute the agent's policy.
            action = agent.act(state, reward,learning=False)
            # Terminal check.
            if state.is_terminal():
                if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    # experiment.add_experience(agent, state, action, 0, state, time_taken = time.clock()-step_start)
                    experiment.add_experience(agent, state, action, 0, state,
                                              time_taken=time.perf_counter() - step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)
            agent.episode_count[episode][state][action] += 1
            agent.episode_reward[episode] += reward
            return_per_episode[episode - 1] += reward

            # Track value.
            value += reward * gamma ** step
            dicounted_return_per_episode[episode - 1] += reward * gamma ** step

            # Record the experience.
            if experiment is not None: # 不会进去的分支。
                reward_to_track = mdp.get_gamma() ** (
                            step + 1 + episode * steps) * reward if track_disc_reward else reward
                reward_to_track = round(reward_to_track, 5)
                # experiment.add_experience(agent, state, action, reward_to_track, next_state, time_taken=time.clock() - step_start)
                experiment.add_experience(agent, state, action, reward_to_track, next_state,
                                          time_taken=time.perf_counter() - step_start)

            if next_state.is_terminal():  #
                success = True
                if reset_at_terminal:  # true
                    # record the position of patient, and step:
                    # record_item =
                    should_add_patient = True
                    for patient in patient_found:
                        if patient[0] == next_state.x and patient[1] == next_state.y:
                            should_add_patient = False
                            break
                    if should_add_patient:
                        patient_found.append([next_state.x, next_state.y, total_step])


                    # patient_found.append([next_state.x,next_state.y,total_step])
                    # the visited state is no longer terminal:
                    next_state._is_terminal = False


                    # Reset the MDP.
                    # next_state = mdp.get_init_state()
                    # agent.step_number = 0
                    # mdp.reset()
                # elif resample_at_terminal and step < steps:
                #     agent.step_number = 0
                #     mdp.reset()
                #     return True, steps, return_per_episode, dicounted_return_per_episode

            # Update pointer.
            state = next_state
            total_step += 1

        # A final update.
        # action = agent.act(state, reward)

        # Process experiment info at end of episode.
        # if experiment is not None:
        #     experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        # mdp.reset()
        # agent.end_of_episode()

        if verbose:
            print("\n")

    # Get the trajectory which has the maxmium reward.
    # max_r = float("-inf")
    # best_e = 0

    # when task finished, calculate collected s,a for reward shaping.
    # for episode in range(1, episodes + 1):
    #     if agent.episode_reward[episode] > max_r:
    #         max_r = agent.episode_reward[episode]  # 这个episode(是这100帧)采样到的最大 accumulated reward.
    #         best_e = episode  # 记录这个回合的序号。

    # for x in agent.episode_count[best_e]:  # 只记录表现最好的那个回合的s,a
    #     for y in agent.episode_count[best_e][x]:
    #         agent.count_sa[x][y] += agent.episode_count[best_e][x][y]
    #         agent.count_s[x] += agent.episode_count[best_e][x][y]
            # agent.count +=agent.episode_count[best_e][x][y]
    # agent.episode_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # 记录完之后重置
    # agent.episode_reward = defaultdict(lambda: 0)

    # Process that learning instance's info at end of learning.
    # if experiment is not None:
    #     experiment.end_of_instance(agent)

    # return False, steps, return_per_episode, dicounted_return_per_episode,success
    return patient_found
def test_strategy_on_mdp_breadth(agent, mdp, episodes, steps, experiment=None, verbose=False, track_disc_reward=False,
                            reset_at_terminal=False, resample_at_terminal=False,task_success_list=None):
    '''
    Summary:
        Main loop of a single MDP experiment.  跑一个task.

    Returns:
        (tuple): (bool:reached terminal, int: num steps taken, float: cumulative discounted reward)
    '''
    if reset_at_terminal and resample_at_terminal:
        raise ValueError(
            "(simple_rl) ExperimentError: Can't have reset_at_terminal and resample_at_terminal set to True.")
    success = False
    value = 0
    gamma = mdp.get_gamma()
    return_per_episode = [0] * episodes
    dicounted_return_per_episode = [0] * episodes


    total_step = 0
    patient_found = []
    # For each episode.
    for episode in range(1, episodes + 1):
        '''
        把200,000分成 200 episodes x 100 steps原因是:每隔一个episode就把agent就更新anneal.
        '''

        # if verbose:
        #     # Print episode numbers out nicely.
        #     sys.stdout.write("\tEpisode %s of %s" % (episode, episodes))
        #     sys.stdout.write("\b" * len("\tEpisode %s of %s" % (episode, episodes)))
        #     sys.stdout.flush()

        # Compute initial state/reward.
        state = mdp.get_init_state()
        # print('breadth','initiate state:',state.x,state.y)
        reward = 0
        # episode_start_time = time.clock()
        # episode_start_time = time.perf_counter()

        # Extra printing if verbose.
        # if verbose:
        #     print()
        #     sys.stdout.flush()
        #     prog_bar_len = _make_step_progress_bar()

        for step in range(1, steps + 1):
            # print(step)
            # if verbose and int(prog_bar_len * float(step) / steps) > int(prog_bar_len * float(step - 1) / steps):
            #     _increment_bar()

            # step time
            # step_start = time.clock()
            step_start = time.perf_counter()

            # # Compute the agent's policy.
            action = agent.act(state, reward,learning=False)
            # Terminal check.
            if state.is_terminal():
                if episodes == 1 and not reset_at_terminal and experiment is not None and action != "terminate":
                    # Self loop if we're not episodic or resetting and in a terminal state.
                    # experiment.add_experience(agent, state, action, 0, state, time_taken = time.clock()-step_start)
                    experiment.add_experience(agent, state, action, 0, state,
                                              time_taken=time.perf_counter() - step_start)
                    continue
                break

            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)

            total_step += 1

            agent.episode_count[episode][state][action] += 1
            agent.episode_reward[episode] += reward
            return_per_episode[episode - 1] += reward

            # Track value.
            value += reward * gamma ** step
            dicounted_return_per_episode[episode - 1] += reward * gamma ** step

            # Record the experience.
            # if experiment is not None: # 不会进去的分支。
            #     reward_to_track = mdp.get_gamma() ** (
            #                 step + 1 + episode * steps) * reward if track_disc_reward else reward
            #     reward_to_track = round(reward_to_track, 5)
            #     experiment.add_experience(agent, state, action, reward_to_track, next_state, time_taken=time.clock() - step_start)
            #     experiment.add_experience(agent, state, action, reward_to_track, next_state,
            #                               time_taken=time.perf_counter() - step_start)

            if next_state.is_terminal():  #
                success = True
                if reset_at_terminal:  # true
                    # Reset the MDP.
                    # check if the position duplicate with the patient found before.
                    should_add_patient = True
                    for patient in patient_found:
                        if patient[0] == next_state.x and patient[1] == next_state.y:
                            should_add_patient = False
                            break
                    if should_add_patient:
                        patient_found.append([next_state.x,next_state.y,total_step])


                    next_state = mdp.get_init_state()
                    agent.step_number = 0
                    mdp.reset()
                # elif resample_at_terminal and step < steps:
                #     agent.step_number = 0
                #     mdp.reset()
                #     return True, steps, return_per_episode, dicounted_return_per_episode

            # Update pointer.
            state = next_state


        # A final update.
        action = agent.act(state, reward)

        # Process experiment info at end of episode.
        # if experiment is not None:
        #     experiment.end_of_episode(agent)

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        # if verbose:
        #     print("\n")

    # Get the trajectory which has the maxmium reward.
    # max_r = float("-inf")
    # best_e = 0

    # when task finished, calculate collected s,a for reward shaping.
    # for episode in range(1, episodes + 1):
    #     if agent.episode_reward[episode] > max_r:
    #         max_r = agent.episode_reward[episode]  # 这个episode(是这100帧)采样到的最大 accumulated reward.
    #         best_e = episode  # 记录这个回合的序号。

    # for x in agent.episode_count[best_e]:  # 只记录表现最好的那个回合的s,a
    #     for y in agent.episode_count[best_e][x]:
    #         agent.count_sa[x][y] += agent.episode_count[best_e][x][y]
    #         agent.count_s[x] += agent.episode_count[best_e][x][y]
            # agent.count +=agent.episode_count[best_e][x][y]
    # agent.episode_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))  # 记录完之后重置
    # agent.episode_reward = defaultdict(lambda: 0)

    # Process that learning instance's info at end of learning.
    # if experiment is not None:
    #     experiment.end_of_instance(agent)

    # return False, steps, return_per_episode, dicounted_return_per_episode,success
    return patient_found
def run_single_belief_agent_on_pomdp(belief_agent, pomdp, episodes, steps, experiment=None, verbose=False,
                                     track_disc_reward=False, reset_at_terminal=False, resample_at_terminal=False):
    '''

    Args:
        belief_agent:
        pomdp:
        episodes:
        steps:
        experiment:
        verbose:
        track_disc_reward:
        reset_at_terminal:
        resample_at_terminal:

    Returns:

    '''
    pass


def _make_step_progress_bar():
    '''
    Summary:
        Prints a step progress bar for experiments.

    Returns:
        (int): Length of the progress bar (in characters).
    '''
    progress_bar_width = 20
    sys.stdout.write("\t\t[%s]" % (" " * progress_bar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progress_bar_width + 1))  # return to start of line, after '['
    return progress_bar_width


def _increment_bar():
    sys.stdout.write("-")
    sys.stdout.flush()


def evaluate_agent(agent, mdp, instances=10):
    '''
    Args:
        agent (simple_rl.Agent)
        mdp (simple_rl.MDP)
        instances (int)

    Returns:
        (float): Avg. cumulative discounted reward.
    '''
    total = 0.0
    steps = int(1 / (1 - mdp.get_gamma()))
    for i in range(instances):
        _, _, val = run_single_agent_on_mdp(agent, mdp, episodes=1, steps=steps)
        total += val
        # Reset the agent.
        agent.reset()
        mdp.end_of_instance()

    return total / instances


def choose_mdp(mdp_name, env_name="Asteroids-v0"):
    '''
    Args:
        mdp_name (str): one of {gym, grid, chain, taxi, ...}
        gym_env_name (str): gym environment name, like 'CartPole-v0'

    Returns:
        (MDP)
    '''

    # Other imports
    from simple_rl.tasks import ChainMDP, GridWorldMDP, FourRoomMDP, TaxiOOMDP, RandomMDP, PrisonersDilemmaMDP, \
        RockPaperScissorsMDP, GridGameMDP

    # Taxi MDP.
    agent = {"x": 1, "y": 1, "has_passenger": 0}
    passengers = [{"x": 4, "y": 3, "dest_x": 2, "dest_y": 2, "in_taxi": 0}]
    walls = []
    if mdp_name == "gym":
        # OpenAI Gym MDP.
        try:
            from simple_rl.tasks.gym.GymMDPClass import GymMDP
        except:
            raise ValueError("(simple_rl) Error: OpenAI gym not installed.")
        return GymMDP(env_name, render=True)
    else:
        return {"grid": GridWorldMDP(5, 5, (1, 1), goal_locs=[(5, 3), (4, 1)]),
                "four_room": FourRoomMDP(),
                "chain": ChainMDP(5),
                "taxi": TaxiOOMDP(10, 10, slip_prob=0.0, agent=agent, walls=walls, passengers=passengers),
                "random": RandomMDP(num_states=40, num_rand_trans=20),
                "prison": PrisonersDilemmaMDP(),
                "rps": RockPaperScissorsMDP(),
                "grid_game": GridGameMDP(),
                "multi": {0.5: RandomMDP(num_states=40, num_rand_trans=20),
                          0.5: RandomMDP(num_states=40, num_rand_trans=5)}}[mdp_name]


def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp", type=str, nargs='?', help="Select the mdp. Options: {atari, grid, chain, taxi}")
    parser.add_argument("-env", type=str, nargs='?', help="Select the Gym environment.")
    args = parser.parse_args()

    # Fix variables based on options.
    task = args.mdp if args.mdp else "grid"
    env_name = args.env if args.env else "CartPole-v0"

    return task, env_name


def main():
    # Command line args.
    task, rom = parse_args()

    # Setup the MDP.
    mdp = choose_mdp(task, rom)
    actions = mdp.get_actions()
    gamma = mdp.get_gamma()

    # Setup agents.
    from simple_rl.agents import RandomAgent, QLearningAgent

    random_agent = RandomAgent(actions)
    qlearner_agent = QLearningAgent(actions, gamma=gamma, explore="uniform")
    agents = [qlearner_agent, random_agent]

    # Run Agents.
    if isinstance(mdp, MarkovGameMDP):
        # Markov Game.
        agents = {qlearner_agent.name: qlearner_agent, random_agent.name: random_agent}
        play_markov_game(agents, mdp, instances=100, episodes=1, steps=500)
    else:
        # Regular experiment.
        run_agents_on_mdp(agents, mdp, instances=50, episodes=1, steps=2000)


if __name__ == "__main__":
    main()
