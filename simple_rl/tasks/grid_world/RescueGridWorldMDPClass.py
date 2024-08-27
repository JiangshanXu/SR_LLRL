''' GridWorldMDPClass.py: Contains the GridWorldMDP class. '''

# Python imports.
from __future__ import print_function
import random
import sys
import os
import numpy as np

# Other imports.
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState

# Fix input to cooperate with python 2 and 3.
try:
   input = raw_input
except NameError:
   pass

class RescueGridWorldMDP(MDP):
    ''' Class for a Grid World MDP '''

    # Static constants.
    ACTIONS = ["up", "down", "left", "right"]

    def __init__(self,
                width=19,
                height=19,
                # init_loc=(1,1),
                # rand_init=False,
                # goal_locs=[(5,3)],
                # lava_locs=[()],
                # walls=[],
                # is_goal_terminal=True,
                gamma=0.99,
                # init_state=None,
                slip_prob=0.0,
                step_cost=0.0,
                # lava_cost=0.01,
                name="Rescue_Grid_World",
                obs_num = 100,
                 patient_count = 3
                 ):
        '''
        Args:
            height (int) grid height
            width (int) grid width
            init_loc (tuple: (int, int))
            goal_locs (list of tuples: [(int, int)...])
            lava_locs (list of tuples: [(int, int)...]): These locations return -1 reward.
        '''
        self.patient_count = patient_count
        # Setup init location.
        # self.rand_init = rand_init
        # if rand_init:
        #     init_loc = random.randint(1, width), random.randint(1, height)
        #     while init_loc in walls:
        #         init_loc = random.randint(1, width), random.randint(1, height)
        # self.init_loc = init_loc
        # init_state = GridWorldState(init_loc[0], init_loc[1]) if init_state is None or rand_init else init_state
        self.start_points = [
            (0, 19), (12, 19), (15, 19), (4, 18), (17, 18)
        ]
        # self.goals = [
        #     (18, 2), (2, 3), (14, 1)
        # ]
        # random generate between (2,2) and (18,5):
        # self.goals = [
        #     (random.randint(2, 18), random.randint(2, 5)) for _ in range(3)
        # ]
        # set goals according to patient count :
        self.goals = [(random.randint(2, 18), random.randint(2, 5)) for _ in range(self.patient_count)]
        print('goals:', self.goals)
        self.goal_locs = self.goals
        # random choose one:
        init_loc = random.choice(self.start_points)
        init_state = GridWorldState(init_loc[0], init_loc[1])
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])

        self.obs_num = obs_num

        MDP.__init__(self, RescueGridWorldMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

        if type(self.goal_locs) is not list:
            raise ValueError("(simple_rl) GridWorld Error: argument @goal_locs needs to be a list of locations. For example: [(3,3), (4,3)].")
        self.step_cost = step_cost
        # self.lava_cost = lava_cost
        self.walls = []
        self.walls = self.create_walls()


        self.width = width
        self.height = height
        # self.goal_locs = goal_locs

        # self.is_goal_terminal = is_goal_terminal
        # is goal terminal become a list:
        self.is_goal_terminal = [False] * len(self.goals)
        self.is_all_goals_terminal = False

        self.slip_prob = slip_prob  # this is zero, meaning that there is no slip in the environment.

        self.name = name
        # self.lava_locs = lava_locs

    def create_walls(self):
        '''
        random generate walls.
        '''
        random_obs = [(random.randint(0, 19), random.randint(0, 16)) for _ in range(self.obs_num)]
        for obs in random_obs:
            if obs in self.goals:
                random_obs.remove(obs)
        # self.walls.extend(random_obs)
        return random_obs
        # print("walls: ", self.walls)


    def _reward_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)
            if reach goal state then present good reward like 10, other wise -1.

        Returns
            (float)
        '''
        if self._is_goal_state_action(state, action):
            return 2.0 - self.step_cost
        # elif (int(state.x), int(state.y)) in self.lava_locs:
        #     return -self.lava_cost
        else:
            return 0 - self.step_cost

    def _is_goal_state_action(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns:
            true if this s,a lead to target s the first time, the first time, the first time.
            (bool): True iff the state-action pair send the agent to the goal state the first time, the first time.
        '''
        # if (state.x, state.y) in self.goal_locs and self.is_goal_terminal:
        #     # Already at terminal.
        #     return False

        # if the goal has been visited, then it will return false.
        if (state.x, state.y) in self.goal_locs:
            # get index:
            goal_index = self.goal_locs.index((state.x, state.y))
            if self.is_goal_terminal[goal_index]:
                return False

        if action == "left" and (state.x - 1, state.y) in self.goal_locs:
            # get index:
            goal_index = self.goal_locs.index((state.x - 1, state.y))
            # if self.is_goal_terminal[goal_index]:
            #     return False
            self.is_goal_terminal[goal_index] = True
            return True
        elif action == "right" and (state.x + 1, state.y) in self.goal_locs:
            goal_index = self.goal_locs.index((state.x + 1, state.y))
            # if self.is_goal_terminal[goal_index]:
            #     return False
            self.is_goal_terminal[goal_index] = True
            return True
        elif action == "down" and (state.x, state.y - 1) in self.goal_locs:
            goal_index = self.goal_locs.index((state.x, state.y - 1))
            # if self.is_goal_terminal[goal_index]:
            #     return False
            self.is_goal_terminal[goal_index] = True
            return True
        elif action == "up" and (state.x, state.y + 1) in self.goal_locs:
            goal_index = self.goal_locs.index((state.x, state.y + 1))
            # if self.is_goal_terminal[goal_index]:
            #     return False
            self.is_goal_terminal[goal_index] = True
            return True
        else:
            return False

    def _transition_func(self, state, action):
        '''
        Args:
            state (State)
            action (str)

        Returns
            (State)
        '''
        if state.is_terminal():
            return state

        r = random.random()
        if self.slip_prob > r:
            # Flip dir.
            if action == "up":
                action = random.choice(["left", "right"])
            elif action == "down":
                action = random.choice(["left", "right"])
            elif action == "left":
                action = random.choice(["up", "down"])
            elif action == "right":
                action = random.choice(["up", "down"])

        if action == "up" and state.y < self.height and not self.is_wall(state.x, state.y + 1):
            next_state = GridWorldState(state.x, state.y + 1)
        elif action == "down" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state = GridWorldState(state.x, state.y - 1)
        elif action == "right" and state.x < self.width and not self.is_wall(state.x + 1, state.y):
            next_state = GridWorldState(state.x + 1, state.y)
        elif action == "left" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state = GridWorldState(state.x - 1, state.y)
        else:
            next_state = GridWorldState(state.x, state.y)

        if (next_state.x, next_state.y) in self.goal_locs:
            # index:
            goal_index = self.goal_locs.index((next_state.x, next_state.y))
            if not self.is_goal_terminal[goal_index]:
                self.is_goal_terminal[goal_index] = True

                if all(self.is_goal_terminal):
                    self.is_all_goals_terminal = True
            print('set as terminal,the next state is:', next_state.x, next_state.y)
            next_state.set_terminal(True)

        return next_state

    def is_wall(self, x, y):
        '''
        Args:
            x (int)
            y (int)

        Returns:
            (bool): True iff (x,y) is a wall location.
        '''

        return (x, y) in self.walls
    def reset(self):
        '''
        reset walls and current state.
        '''
        # set current state:
        # random generate init state:
        init_loc = random.choice(self.start_points)
        # set to cur_state:
        self.cur_state = GridWorldState(init_loc[0], init_loc[1])

        # reset walls:
        # do not reset walls because reset walls will change the environment.
        # self.walls = self.create_walls()


    def __str__(self):
        return self.name + "_h-" + str(self.height) + "_w-" + str(self.width)

    def get_goal_locs(self):
        return self.goal_locs

    # def get_lava_locs(self):
    #     return self.lava_locs

    def visualize_policy(self, policy):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state

        action_char_dict = {
            "up":"^",       #u"\u2191",
            "down":"v",     #u"\u2193",
            "left":"<",     #u"\u2190",
            "right":">",    #u"\u2192"
        }

        mdpv.visualize_policy(self, policy, _draw_state, action_char_dict)
        input("Press anything to quit")

    def visualize_agent(self, agent):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_agent(self, agent, _draw_state)
        input("Press anything to quit")

    def visualize_value(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_value(self, _draw_state)
        input("Press anything to quit")

    def visualize_learning(self, agent, delay=0.0):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_learning(self, agent, _draw_state, delay=delay)
        input("Press anything to quit")

    def visualize_interaction(self):
        from simple_rl.utils import mdp_visualizer as mdpv
        from simple_rl.tasks.grid_world.grid_visualizer import _draw_state
        mdpv.visualize_interaction(self, _draw_state)
        input("Press anything to quit")

def _error_check(state, action):
    '''
    Args:
        state (State)
        action (str)

    Summary:
        Checks to make sure the received state and action are of the right type.
    '''

    if action not in GridWorldMDP.ACTIONS:
        raise ValueError("(simple_rl) GridWorldError: the action provided (" + str(action) + ") was invalid in state: " + str(state) + ".")

    if not isinstance(state, GridWorldState):
        raise ValueError("(simple_rl) GridWorldError: the given state (" + str(state) + ") was not of the correct class.")

def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
    '''
    Args:
        file_name (str)
        randomize (bool): If true, chooses a random agent location and goal location.
        num_goals (int)
        name (str)

    Returns:
        (GridWorldMDP)

    Summary:
        Builds a GridWorldMDP from a file:
            'w' --> wall
            'a' --> agent
            'g' --> goal
            '-' --> empty
    '''

    if name is None:
        name = file_name.split(".")[0]

    # grid_path = os.path.dirname(os.path.realpath(__file__))
    wall_file = open(os.path.join(os.getcwd(), file_name))
    wall_lines = wall_file.readlines()

    # Get walls, agent, goal loc.
    num_rows = len(wall_lines)
    num_cols = len(wall_lines[0].strip())
    empty_cells = []
    agent_x, agent_y = 1, 1
    walls = []
    goal_locs = []
    lava_locs = []

    for i, line in enumerate(wall_lines):
        line = line.strip()
        for j, ch in enumerate(line):
            if ch == "w":
                walls.append((j + 1, num_rows - i))
            elif ch == "g":
                goal_locs.append((j + 1, num_rows - i))
            elif ch == "l":
                lava_locs.append((j + 1, num_rows - i))
            elif ch == "a":
                agent_x, agent_y = j + 1, num_rows - i
            elif ch == "-":
                empty_cells.append((j + 1, num_rows - i))

    if goal_num is not None:
        goal_locs = [goal_locs[goal_num % len(goal_locs)]]

    if randomize:
        agent_x, agent_y = random.choice(empty_cells)
        if len(goal_locs) == 0:
            # Sample @num_goals random goal locations.
            goal_locs = random.sample(empty_cells, num_goals)
        else:
            goal_locs = random.sample(goal_locs, num_goals)

    if len(goal_locs) == 0:
        goal_locs = [(num_cols, num_rows)]

    return GridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob)

    def reset(self):
        if self.rand_init:
            init_loc = random.randint(1, width), random.randint(1, height)
            self.cur_state = GridWorldState(init_loc[0], init_loc[1])
        else:
            self.cur_state = copy.deepcopy(self.init_state)

def main():
    grid_world = GridWorldMDP(5, 10, (1, 1), (6, 7))

    grid_world.visualize()

if __name__ == "__main__":
    main()
