import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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


class ActorCritic(nn.Module):
    def __init__(self, env, bounded: bool = False, gamma: float = 0.99):
        super().__init__()
        self._actor = Actor(env)
        self._critic1 = QNetwork(env)
        self._critic2 = QNetwork(env)
        self.bounded = bounded
        self.gamma = gamma

    @property
    def action_scale(self):
        return self._actor.action_scale

    def forward_critic1(self, x, a):
        val = self._critic1(x, a)
        if self.bounded:
            val = torch.sigmoid(val) / (1 - self.gamma)
        return val

    def forward_critic2(self, x, a):
        val = self._critic2(x, a)
        if self.bounded:
            val = torch.sigmoid(val) / (1 - self.gamma)
        return val

    def forward_actor(self, x):
        return self._actor(x)

    def get_q_value(self, x, a):
        q1 = self.forward_critic1(x, a)
        q2 = self.forward_critic2(x, a)
        return torch.min(q1, q2)[0]

    def get_value(self, x):
        a = self.forward_actor(x)
        return self.get_q_value(x, a)

    def get_critic_parameters(self):
        return list(self._critic1.parameters()) + list(self._critic2.parameters())

    def get_actor_parameters(self):
        return self._actor.parameters()
