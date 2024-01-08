import copy
import functools
import time
from pettingzoo import ParallelEnv
import gymnasium as gym
import math
import numpy as np
import sys


def env(
    render_mode=None,
):
    env = parallel_env()
    return env


class parallel_env(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode=None,
    ):
        self.render_mode = render_mode

        self.agents = [0, 1]
        self.possible_agents = self.agents[:]

        action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_spaces = {i: action_space for i in self.agents}

        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(625,), dtype=np.float32
        )
        self.observation_spaces = {i: observation_space for i in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        """
        Maze:
        g = goal = 1
        x = wall = -1
        s = agent start = 2
        
        [0, 0, g, 0, 0]
        [0, x, 0, x, 0]
        [0, x, 0, x, 0]
        [0, x, 0, x, 0]
        [0, s, 0, s, 0]
        """
        obs, info = {}, {}
        
        self.positions = [
            [4, 1],
            [4, 3],
        ]
        
        self.goal = [0, 2]
        
        self.walls = [
            [1, 1],
            [2, 1],
            [3, 1],
            [1, 3],
            [2, 3],
            [3, 3],
        ]
        

        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            info[agent] = {}

        return obs, info

    def step(self, actions):
        
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        
        # Save old positions
        old_positions = copy.deepcopy(self.positions)
        
        # Move agents
        for agent in self.agents:
            self.move_agent(agent, actions[agent])
            
        self.collision = False
            
        # Check if agents are on the same position
        if self.positions[0] == self.positions[1] and self.positions[0] != self.goal:
            # Reset positions
            self.positions = old_positions
            self.collision = True
            
        # Check if agents are on "walls"
        for agent in self.agents:
            for wall in self.walls:
                if self.positions[agent] == wall:
                    # Reset positions
                    self.positions[agent] = old_positions[agent]
                    break
            
            
        goal_reached = False    
        # Check if both agents are on the goal
        if self.positions[0] == self.goal and self.positions[1] == self.goal:
            goal_reached = True
            
        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.get_rew(agent)
            terminated[agent] = goal_reached
            truncated[agent] = False
            info[agent] = {}

        return obs, rew, terminated, truncated, info
    
    def get_direction(self, action):
        if action < -0.5:
            # Move left
            return [0, -1]
        elif action < 0:
            # Move down
            return [1, 0]
        elif action < 0.5:
            # Move right
            return [0, 1]
        else:
            # Move up
            return [-1, 0]
    
    def move_agent(self, agent, action):
        # Get position
        position = self.positions[agent]
        
        # If agent is on goal, don't move
        if position == self.goal:
            return
        
        # Convert action to direction
        direction = self.get_direction(action)
        
        # Calculate new position
        new_position = [position[0] + direction[0], position[1] + direction[1]]
        
        # Clip new position
        new_position[0] = np.clip(new_position[0], 0, 4)
        new_position[1] = np.clip(new_position[1], 0, 4)
        
        # Set new position
        self.positions[agent] = new_position
        
    
    def state_to_one_hot(self, state):
        one_hot_state = np.zeros((5, 5))
        one_hot_state[state[0]][state[1]] = 1
        return one_hot_state.flatten()
    
    def combine_states(self, state1, state2, grid_size):
        # Find the indices of the active states in the two vectors
        index1 = np.argmax(state1)
        index2 = np.argmax(state2)

        # Calculate the index in the combined vector
        combined_index = index1 * grid_size * grid_size + index2

        # Create a new one-hot encoded vector
        combined_state = np.zeros(grid_size * grid_size * grid_size * grid_size)
        combined_state[combined_index] = 1

        return combined_state

    def get_obs(self, agent):
        # Get the position and state of the agent
        position = self.positions[agent]
        selfState = self.state_to_one_hot(position)
        
        # Get position and state of other agent
        other_agent = 1 - agent
        other_position = self.positions[other_agent]
        otherState = self.state_to_one_hot(other_position)

        return self.combine_states(selfState, otherState, 5)
        
    def get_rew(self, agent):
        if self.collision:
            return -100
        
        if self.positions[agent] == self.goal:
            return 1
        else:
            return -1
        
    def render(self):
        # Print maze to console
        for x in range(5):
            for y in range(5):
                if [x, y] == self.goal:
                    print("g", end="")
                elif [x, y] in self.walls:
                    print("x", end="")
                elif [x, y] in self.positions:
                    print("s", end="")
                else:
                    print(" ", end="")
            print("")
        print("----------------")
        time.sleep(0.2)