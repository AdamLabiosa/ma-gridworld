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
        self.global_time = 0

        self.agents = [0, 1]
        self.possible_agents = self.agents[:]

        self.state = None

        action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.action_spaces = {i: action_space for i in self.agents}

        observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_spaces = {i: observation_space for i in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, return_info=False, options=None, **kwargs):
        self.global_time = 0

        obs, info = {}, {}

        # Goal output
        self.state = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            info[agent] = {}

        return obs, info

    def step(self, actions):
        self.global_time += 1

        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}

        for agent in self.agents:
            obs[agent] = self.get_obs(agent)
            rew[agent] = self.get_rew(agent, actions[agent])
            terminated[agent] = self.global_time > 100
            truncated[agent] = False
            info[agent] = {}

        return obs, rew, terminated, truncated, info

    def get_obs(self, agent):        
        return np.array([self.state[agent], agent == 0, agent == 1])

    def get_rew(self, agent, action):
        # Get distance to goal
        goal = self.state[agent]

        return -np.abs(goal - action[0], dtype=np.float32)
