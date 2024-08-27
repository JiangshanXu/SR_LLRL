###########################################################################################
# Implementation of Q-Learning Agent with Lifetime Reward Shaping Function
# Author for codes: Chu Kun(kun_chu@outlook.com)
# Reference: https://github.com/Kchu/LifelongRL
###########################################################################################

# Python imports.
import random
import numpy
import time
import copy
import math
from collections import defaultdict
import random
# import gym
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Other imports.
from simple_rl.agents.AgentClass import Agent
from simple_rl.planning.ValueIterationClass import ValueIteration
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

def sigmod(x):
    return 1 / (1 + numpy.exp(-x))

class DQLearningAgent(Agent):
    ''' Implementation for a CBRS Q Learning Agent '''

    def __init__(self, actions, name="D-Q-learning", init_q=None, alpha=0.05, gamma=0.99, epsilon=0.1, explore="uniform", anneal=False,
                 default_q=1.0/(1.0-0.99)
                 ,state_dim=2
                 ):
        # state_n, action_n,
        '''
        Args:
            actions (list): Contains strings denoting the actions. # ['up', 'down', 'left', 'right']
            name (str): Denotes the name of the agent.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration term.
            explore (str): One of {softmax, uniform}. Denotes explore policy.
        '''
        name_ext = "-" + explore if explore != "uniform" else ""
        Agent.__init__(self, name=name + name_ext, actions=actions, gamma=gamma)

        # Set/initialize parameters and other relevant classwide data
        self.alpha, self.alpha_init = alpha, alpha
        self.epsilon, self.epsilon_init = epsilon, epsilon

        self.step_number = 0
        self.anneal = anneal

        self.default_q = default_q
        # self.init_q = defaultdict(lambda : defaultdict(lambda: self.default_q)) if init_q is None else init_q
        # self.default_q_func = copy.deepcopy(self.init_q)
        # self.q_func = copy.deepcopy(self.default_q_func)
        self.q_func = None
        self.action_dim = len(actions)
        self.state_dim = state_dim
        self.learning_rate = 0.01
        self.target_update_freq = 10
        self.gamma = gamma
        self.epsilon = epsilon
        # self.update_count = 0
        self.count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set to cpu:
        self.device = torch.device("cpu")

        self.buffer_size = 10000
        self.minimal_buffer_size = 500
        self.batch_size  = 64
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.q_net = Qnet(state_dim, 128, self.action_dim)
        self.target_q_net = Qnet(state_dim, 128, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)



        # LRS setting
        self.count_sa = defaultdict(lambda : defaultdict(lambda: 0))
        self.count_s= defaultdict(lambda : 0)
        self.episode_count = defaultdict(lambda : defaultdict(lambda: defaultdict(lambda: 0)))
        self.episode_reward = defaultdict(lambda: 0)
        self.reward_sa = defaultdict(lambda : defaultdict(lambda: 0))

        # Choose explore type.
        self.explore = explore

        self.task_number = 0
        # self.num_sample_tasks = 100

    # --------------------------------
    # ---- CENTRAL ACTION METHODS ----
    # --------------------------------
    def q_func_reset(self):
        '''
        reset the q_func and q_target
        '''
        self.q_net = Qnet(self.state_dim, 128, self.action_dim)
        self.target_q_net = Qnet(self.state_dim, 128, self.action_dim)


    def dqn_update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def act(self, state, reward, explore=True, learning=True):
        '''
        Args:
            state (State)
            reward (float)
        Summary:
            The central method called during each time step.
            Retrieves the action according to the current policy
            and performs updates given (s=self.prev_state,
            a=self.prev_action, r=reward, s'=state)
        '''
        done = state.is_terminal()
        state = state.get_data()
        if self.prev_state is not None:
            self.replay_buffer.add(self.prev_state, self.prev_action, reward, state, done)


        if learning and self.replay_buffer.size() > self.minimal_buffer_size:
            # self.update(self.prev_state, self.prev_action, reward, state)
            b_state, b_action, b_reward, b_next_state, b_done = self.replay_buffer.sample(self.batch_size)
            # transition_dict = {'state':b_state, 'action':b_action, 'reward':b_reward, 'next_state':b_next_state, 'done':b_done}
            transition_dict = {'states':b_state, 'actions':b_action, 'rewards':b_reward, 'next_states':b_next_state, 'dones':b_done}
            self.dqn_update(transition_dict)
        # if explore:
        #     if self.explore == "softmax":
        #         # Softmax exploration
        #         action = self.soft_max_policy(state)
        #     else:
        #         # Uniform exploration
        #         action = self.epsilon_greedy_q_policy(state)
        # else:
        #     action = self.get_max_q_action(state)
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
            # to index:
            action = self.actions.index(action)

        else:
            # state = state.get_data()
            # to torch:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()


        self.prev_state = state
        self.prev_action = action
        self.step_number += 1

        # Anneal params.
        # if learning and self.anneal:
        #     self._anneal()

        # index to str when returned:
        action = self.actions[action]

        return action

    def epsilon_greedy_q_policy(self, state):
        '''
        Args:
            state (State)
        Returns:
            (str): action.
        '''
        assert 0, "should never be triggered"
        # Policy: Epsilon of the time explore, otherwise, greedyQ.
        if numpy.random.random() > self.epsilon:
            # Exploit.
            action = self.get_max_q_action(state)
        else:
            # Explore
            action = numpy.random.choice(self.actions)

        return action

    def soft_max_policy(self, state):
        '''
        Args:
            state (State): Contains relevant state information.
        Returns:
            (str): action.
        '''
        assert 0,"should never be triggered"
        return numpy.random.choice(self.actions, 1, p=self.get_action_distr(state))[0]

    # ---------------------------------
    # ---- Q VALUES AND PARAMETERS ----
    # ---------------------------------

    def update(self, state, action, reward, next_state):
        '''
        Args:
            state (State)
            action (str)
            reward (float)
            next_state (State)
        Summary:
            Updates the internal Q Function according to the Bellman Equation. (Classic Q Learning update)
        '''
        # If this is the first state, just return.
        assert 0,"should never be triggerd"
        if state is None:
            self.prev_state = next_state
            return

        if state.is_terminal():
            # If the state is terminal we set the Q values to 0
            for a in self.actions:
                self.q_func[state][a] = 0.0
            # print("State is terminal!")
            # print(self.q_func[state])
            return
        
        if next_state.is_terminal():
            # If the state is terminal we set the Q values to 0
            for a in self.actions:
                self.q_func[state][a] = 0.0
            # print("next_state is terminal!")
            # print(self.q_func[state])
            return

        # Update the Q Function.
        max_q_curr_state = self.get_max_q_value(next_state)
        prev_q_val = self.get_q_value(state, action)
        f_reward = self.reward_sa[state][action]
        reward = reward + f_reward
        self.q_func[state][action] = (1 - self.alpha) * prev_q_val + self.alpha * (reward + self.gamma*max_q_curr_state)

    def _anneal(self): 
        self.alpha = self.alpha * self.tau
        # Taken from "Note on learning rate schedules for stochastic optimization, by Darken and Moody (Yale)":
        # self.alpha = self.alpha_init / (1.0 +  (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )
        # self.epsilon = self.epsilon_init / (1.0 + (self.step_number / 200.0)*(self.episode_number + 1) / 2000.0 )

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)
        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # assert(not state.is_terminal())
        # # Grab random initial action in case all equal
        # best_action = random.choice(self.actions)
        # max_q_val = float("-inf")
        # shuffled_action_list = self.actions[:]
        # random.shuffle(shuffled_action_list)
        #
        # # Find best action (action w/ current max predicted Q value)
        # for action in shuffled_action_list:
        #     q_s_a = self.get_q_value(state, action)
        #     q_s_a = q_s_a
        #     if q_s_a > max_q_val:
        #         max_q_val = q_s_a
        #         best_action = action
        #
        # return max_q_val, best_action
        # use q-net:
        assert not state.is_terminal()
        state = state.get_data()
        return self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).max().item(), self.actions[self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).argmax().item()]

    # compute LRS reward function
    def _compute_count_reward(self):
        for x in self.count_sa:
            for y in self.count_sa[x]:
                self.reward_sa[x][y] = (1- self.gamma) * ((self.count_sa[x][y] / self.count_s[x])) * self.default_q

    def get_max_q_action(self, state):
        '''
        Args:
            state (State)
        Returns:
            (str): denoting the action with the max q value in the given @state.
        '''
        # return self._compute_max_qval_action_pair(state)[1]
        # use q-net:
        state = state.get_data()
        return self.actions[self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).argmax().item()]

    def get_max_q_value(self, state):
        '''
        Args:
            state (State)
        Returns:
            (float): denoting the max q value in the given @state.
        '''
        # return self._compute_max_qval_action_pair(state)[0]
        # use q-net:
        state = state.get_data()
        return self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).max().item()
    def get_q_value(self, state, action):
        '''
        Args:
            state (State)
            action (str)
        Returns:
            (float): denoting the q value of the (@state, @action) pair.
        '''
        # return self.q_func[state][action]
        state = state.get_data()
        action = self.actions.index(action)
        # return self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).gather(1, torch.tensor([self.actions.index(action)]).to(self.device)).item()
        return self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).gather(1, torch.tensor([action]).to(self.device)).item()
    def get_action_distr(self, state, beta=0.2):
        '''
        Args:
            state (State)
            beta (float): Softmax temperature parameter.
        Returns:
            (list of floats): The i-th float corresponds to the probability
            mass associated with the i-th action (indexing into self.actions)
        '''
        # all_q_vals = []
        # for i in range(len(self.actions)):
        #     action = self.actions[i]
        #     all_q_vals.append(self.get_q_value(state, action))
        #
        # # Softmax distribution.
        # total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        # softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]
        #
        # return softmax
        # use qnet :
        state = state.get_data()
        all_q_vals = self.q_net(torch.tensor(state, dtype=torch.float).to(self.device)).detach().numpy()
        total = sum([numpy.exp(beta * qv) for qv in all_q_vals])
        softmax = [numpy.exp(beta * qv) / total for qv in all_q_vals]
        return softmax

    def reset(self, mdp=None):
        self.step_number = 0
        self.episode_number = 0
        self._compute_count_reward()  # 每一次reset的时候就会计算reward heuristics : f, 然后就是
        # self.q_func = defaultdict(lambda: defaultdict(lambda: self.default_q))
        # reset q_net and target_q_net
        self.q_net = Qnet(self.state_dim, 128, self.action_dim)
        self.target_q_net = Qnet(self.state_dim, 128, self.action_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        # reset buffer:
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.count = 0

        self.task_number = self.task_number + 1
        Agent.reset(self)

    def end_of_episode(self):
        '''
        Summary:
            Resets the agents prior pointers.
        '''
        # if self.anneal:
        #     self._anneal()
        Agent.end_of_episode(self)

    def set_init_q_function(self, q_func):
        '''
        Function for transferring q function
        '''
        self.default_q_func = copy.deepcopy(q_func)
        self.q_func = copy.deepcopy(self.default_q_func)

    def print_dict(self, dic):
        for x in dic:
            for y in dic[x]:
                print("%.2f" % dic[x][y], end='')
            print("")
