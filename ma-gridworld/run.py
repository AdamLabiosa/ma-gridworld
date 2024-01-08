import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
import supersuit as ss
import argparse

from envs.magrid import env


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--render", action="store_true")

    return parser.parse_args()


def train():
    simple_env = env()
    simple_env = ss.pettingzoo_env_to_vec_env_v1(simple_env)

    envs = ss.concat_vec_envs_v1(
        simple_env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    envs = VecMonitor(envs)

    model = PPO(
        MlpPolicy,
        envs,
        batch_size=128,
        ent_coef=0.01,
        verbose=1,
    )

    model.learn(total_timesteps=250000)
    model.save("ppo_magrid")


def render():
    simple_env = env()
    vec_env = ss.pettingzoo_env_to_vec_env_v1(simple_env)
    model = PPO.load("ppo_magrid")

    obs, info = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = vec_env.step(action)
        vec_env.render()


if __name__ == "__main__":
    args = parse_args()
    if args.train:
        train()
    elif args.render:
        render()
    else:
        print("Please specify either --train or --render")
