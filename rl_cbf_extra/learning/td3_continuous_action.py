# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import abc
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


class SafetyEnv(abc.ABC, gym.Wrapper):
    """TODO: write safety env interface"""

    def __init__(self, env, override_reward=False):
        super().__init__(env)
        self.override_reward = override_reward
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            # "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action: np.ndarray):
        state, reward, _, info = super().step(action)
        is_unsafe = self.is_unsafe(state).item()
        # TODO: overwrite the done condition to match is_unsafe_th
        done = is_unsafe
        if self.override_reward:
            reward = 0.0 if is_unsafe else 1.0
        return state, reward, done, info

    def is_unsafe(self, states: np.ndarray) -> np.ndarray:
        """Numpy wrapper around is_unsafe_th

        Args:
            states: (batch_size, state_dim) array of states

        Returns:
            is_unsafe: (batch_size, 1) float array indicating whether states are unsafe
        """
        return self.is_unsafe_th(torch.from_numpy(states)).numpy()

    @staticmethod
    @abc.abstractmethod
    def is_unsafe_th(states: torch.Tensor) -> torch.Tensor:
        """Return boolean array indicating whether states are unsafe

        Args:
            states: (batch_size, state_dim) array of states

        Returns:
            is_unsafe: (batch_size, 1) float array indicating whether states are unsafe
        """
        raise NotImplementedError

    def sample_states(self, n_states: int) -> np.ndarray:
        """Sample n_states from the environment

        Args:
            n_states: number of states to sample

        Returns:
            states: (n_states, state_dim) array of sampled states
        """
        return np.random.uniform(
            low=-10.0, high=10.0, size=(n_states, self.observation_space.shape[0])
        )


class SafetyWalker2dEnv(SafetyEnv):
    @staticmethod
    def is_unsafe_th(states: torch.Tensor):
        height = states[..., 0]
        angle = states[..., 1]

        height_ok = torch.logical_and(height > 0.8, height < 2.0)
        angle_ok = torch.logical_and(angle > -1.0, angle < 1.0)
        is_safe = torch.logical_and(height_ok, angle_ok)
        return (~is_safe).float().view(-1, 1)

    def reset_to(self, state: np.ndarray):
        new_state = self.reset()
        # TODO: implement this
        return new_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    return args

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # fmt: off   
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-group", type=str, default=None,
        help="the wandb's group name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Hopper-v4",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=25e3,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    # RL-CBF specific arguments
    parser.add_argument("--bounded", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the CBF is bounded")
    parser.add_argument("--supervised", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, the CBF is supervised")
    return parser


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = SafetyWalker2dEnv(env, override_reward=True)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, bounded: bool = False, gamma: float = 0.99):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.bounded = bounded
        self.gamma = gamma

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.bounded:
            # Map into range [0, 1 / (1 - gamma)]
            x = torch.sigmoid(x) / (1 - self.gamma)
        return x

    def predict(self, x: np.ndarray, a: np.ndarray, device: torch.device = "cpu"):
        x = torch.from_numpy(x).to(torch.float32).to(device)
        a = torch.from_numpy(a).to(torch.float32).to(device)
        x = x.view(1, -1)
        a = a.view(1, -1)
        return self.forward(x, a).cpu().detach().numpy()[0]


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

    def predict(self, x: np.ndarray, device: torch.device = "cpu"):
        x = torch.from_numpy(x).to(torch.float32).to(device)
        x = x.view(1, -1)
        return self.forward(x).cpu().detach().numpy()[0]
