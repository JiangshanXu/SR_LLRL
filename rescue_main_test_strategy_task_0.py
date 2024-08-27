#!/usr/bin/env python

###########################################################################################
# Implementation for experimenting with proposed approaches to Lifelong RL, 
# attached to our 2021 IEEE SMC paper 
# "Accelerating lifelong reinforcement learning via reshaping rewards".
# Author for codes: Chu Kun(kun_chu@outlook.com), Abel
# Reference: https://github.com/Kchu/LifelongRL
###########################################################################################

# Python imports.
from collections import defaultdict
import numpy as np
import sys
import copy
import argparse

## Experiment imports. (please import simple_rl in this folder!)
# Basic imports
from utils import make_mdp_distr
from utils import make_rescue_grid_world_distribution
# from simple_rl.mdp import MDP, MDPDistribution
from simple_rl.run_experiments import run_agents_lifelong,run_agents_lifelong_test_strategy
from simple_rl.planning.ValueIterationClass import ValueIteration
# MaxQinit Agents
from agents.MaxQinitQLearningAgentClass import MaxQinitQLearningAgent
from agents.MaxQinitDelayedQAgentClass import MaxQinitDelayedQAgent
# LRS Agents
from agents.LRSQLearningAgentClass import LRSQLearningAgent
from agents.LRSDelayedQAgentClass import LRSDelayedQAgent
# Baselines Agents
from agents.QLearningAgentClass import QLearningAgent
from agents.DelayedQAgentClass import DelayedQAgent

from agents.DQLearningAgentClass import DQLearningAgent

from agents.SRDQLearningAgentClass import SRDQLearningAgent

from agents.multi_LRSQLearningAgentClass import MultiLRSQLearningAgent
from agents.dyna_LRSQLearningAgentClass import DynaLRSQLearningAgent
from agents.sarsa_lambda_LRSQLearningAgentClass import SarsaLambdaLRSQLearningAgent
from agents.sarsa_lambda_2_LRSQLearningAgentClass import SarsaLambda2LRSQLearningAgent

def get_q_func(vi):
    state_space = vi.get_states()
    q_func = defaultdict(lambda: defaultdict(lambda: 0))  # Assuming R>=0
    for s in state_space:
        for a in vi.actions:
            q_s_a = vi.get_q_value(s, a)
            q_func[s][a] = q_s_a
    return q_func


def parse_args():
    '''
    Summary:
        Parse all arguments
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-mdp_class", type=str, default="rescue_grid", nargs='?',
                        help="Choose the mdp type (one of {octo, hall, grid, taxi, four_room}).")
    parser.add_argument("-goal_terminal", type=bool, default=True, nargs='?', help="Whether the goal is terminal.")
    parser.add_argument("-samples", type=int, default=40, nargs='?', help="Number of samples for the experiment.")
    parser.add_argument("-instances", type=int, default=1, nargs='?', help="Number of instances for the experiment.")
    parser.add_argument("-baselines_type", type=str, default="q", nargs='?',
                        help="Type of agents: (q, rmax, delayed-q).")
    args = parser.parse_args()

    return args.mdp_class, args.goal_terminal, args.samples, args.instances, args.baselines_type


def compute_optimistic_q_function(mdp_distr, sample_rate=5):
    '''
    求这个环境的每个s,a的最大值，这个最大值在不同任务间比较。
    Instead of transferring an average Q-value, we transfer the highest Q-value in MDPs so that
    it will not under estimate the Q-value.
    '''
    opt_q_func = defaultdict(lambda: defaultdict(lambda: float("-inf")))  # 初始化一个opt_Q 函数
    for mdp in mdp_distr.get_mdps():  # 多个MDP，都是同一个任务的。
        # Get a vi instance to compute state space.
        vi = ValueIteration(mdp, delta=0.0001, max_iterations=1000, sample_rate=sample_rate)
        iters, value = vi.run_vi()  # 使用遍历迭代的方式来刷新Q值。
        q_func = get_q_func(vi)  # 把算好的价值函数赋值给Q函数。
        for s in q_func:
            for a in q_func[s]:
                opt_q_func[s][a] = max(opt_q_func[s][a], q_func[s][a])
    return opt_q_func  # 求这个mdp的最大值。


def main(open_plot=True):
    # set_experiment_content = "task_num_ablation_0"
    is_ablation = False
    # Environment setting
    pretraining_task_num = 1
    episodes = 200
    steps = 200
    gamma = 0.95
    mdp_size = 11
    vs_task = True
    mdp_class, is_goal_terminal, samples, instance_number, baselines = parse_args()
    obstacle_num = 100
    patient_count = 3
    # Setup multitask setting.
    # mdp_distr = make_mdp_distr(mdp_class=mdp_class, mdp_size=mdp_size, is_goal_terminal=is_goal_terminal, gamma=gamma)
    mdp_distr = make_rescue_grid_world_distribution(obs_num=obstacle_num,patient_count=patient_count)
    val_mdp_distr = make_rescue_grid_world_distribution(obs_num=obstacle_num,patient_count=patient_count)
    print(mdp_distr.get_num_mdps())
    actions = mdp_distr.get_actions()  # ['up', 'down', 'left', 'right']

    # Get optimal q-function for ideal agent.
    opt_q_func = compute_optimistic_q_function(mdp_distr)  # opt_q_func(s,a)代表每个s,a的最大值，这个最大值在不同任务间比较。

    # Get maximum possible value an agent can get in this environment.
    best_v = -100
    for x in opt_q_func:
        for y in opt_q_func[x]:
            best_v = max(best_v, opt_q_func[x][y])  # 算出最大的Q值。从opt_q_func中找到最大的Q值。
    vmax = best_v
    print("Best Vmax =", vmax)

    # Init q-funcion
    vmax_func = defaultdict(lambda: defaultdict(lambda: vmax))  # 初始化一个vmax函数，这个函数的值都是vmax。

    # Different baseline learning algorithms
    if baselines == "q":
        # parameter for q-learning
        eps = 0.1
        lrate = 0.1

        # basic q-learning agent.这个是没有初始化的q表。
        Baseline = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="Baseline")

        # basic q-learning agent with vmax initialization
        # pure_ql_agent_opt = QLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, default_q=vmax, name="VTR")

        # MaxQinit agent
        MaxQinit = MaxQinitQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax,
                                          name="MaxQInit")

        # Ideal agent, ideal agent的Q值表都是算出来的表。
        Ideal = QLearningAgent(actions, init_q=opt_q_func, gamma=gamma, alpha=lrate, epsilon=eps, name="Ideal")

        # SR-LLRL agent
        SR_LLRL = LRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="SR_LLRL")

        # multi_sr_llrl = MultiLRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="multi_SR_LLRL")
        multi_sr_llrl = MultiLRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax,
                                               name="Sarsa(5)_SR")

        dyna_sr_llrl = DynaLRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="dyna_SR")
        # agents
        # DQN = DQLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="DQN", state_dim=2)

        # sr agent:
        # SRDQN = SRDQLearningAgent(actions, gamma=gamma, alpha=lrate, epsilon=eps, name="SRDQN", state_dim=2)

        sarsa_lambda_sr_llrl = SarsaLambdaLRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="sarsa_lambda_SR")
        sarsa_lambda2_sr_llrl = SarsaLambda2LRSQLearningAgent(actions, alpha=lrate, epsilon=eps, gamma=gamma, default_q=vmax, name="sarsa_lambda2_SR")
        # agents = [sarsa_lambda_sr_llrl,multi_sr_llrl,SR_LLRL,dyna_sr_llrl]
        # agents = [sarsa_lambda2_sr_llrl,sarsa_lambda_sr_llrl,multi_sr_llrl,SR_LLRL,dyna_sr_llrl]
        # agents = [SR_LLRL]
        agents = [sarsa_lambda_sr_llrl]
        # only MaxQinit and DQN:
        # agents = [SRDQN,MaxQinit, DQN, SR_LLRL,Baseline]
        alg = "q-learning"

    elif baselines == "delayed-q":
        # parameter for delayed-q
        torelance = 0.001
        min_experience = 5

        # basic delayed-q agent
        Baseline = DelayedQAgent(actions, init_q=vmax_func, gamma=gamma, m=min_experience, epsilon1=torelance,
                                 name="Baseline")

        # MaxQinit agent
        MaxQinit = MaxQinitDelayedQAgent(actions, init_q=vmax_func, default_q=vmax, gamma=gamma, m=min_experience,
                                         epsilon1=torelance, name="MaxQInit")

        # Ideal agent
        Ideal = DelayedQAgent(actions, init_q=opt_q_func, gamma=gamma, m=min_experience, epsilon1=torelance,
                              name="Ideal")

        # ALLRL-RS agent
        SR_LLRL = LRSDelayedQAgent(actions, init_q=vmax_func, gamma=gamma, default_q=vmax, m=min_experience,
                                   epsilon1=torelance, name="SR_LLRL")
        # agents
        agents = [Ideal, MaxQinit, SR_LLRL, Baseline]
        alg = "delayed-q"

    else:
        msg = "Unknown type of agent:" + baselines + ". Use -agent_type (q, rmax, delayed-q)"
        assert False, msg

    # Experiment body
    run_agents_lifelong_test_strategy(agents, mdp_distr, vs_task=vs_task, samples=samples, episodes=episodes, steps=steps,
                        instances=instance_number, reset_at_terminal=is_goal_terminal, alg=alg,
                        track_disc_reward=False, cumulative_plot=False, open_plot=open_plot,env_name=mdp_class,
                        obstacle_num=obstacle_num
                        ,is_ablation=is_ablation,
                        patient_count=patient_count,
                                      pretraining_task_num = pretraining_task_num,
                                      val_mdps=val_mdp_distr
                        )


if __name__ == "__main__":
    open_plot = True
    main(open_plot=open_plot)
