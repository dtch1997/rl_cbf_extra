import torch
import pandas as pd
import torch.nn as nn
import gym

from rl_cbf_extra.learning.td3_continuous_action import Actor, QNetwork


def evaluate_constrain_episode(
    actor: nn.Module,
    qf1: nn.Module,
    qf2: nn.Module,
    env: gym.Env,
    safety_threshold: float = 0,
    render: bool = False,
):
    """Evaluate model on environment for 1 episode"""

    # Roll out model
    state = env.reset()
    done = False
    i = 0

    while not done:
        if render:
            env.render()

        # Take a random action
        action = env.action_space.sample()
        next_qf1_value = qf1.predict(state, action).item()
        next_qf2_value = qf2.predict(state, action).item()
        next_barrier_value = min(next_qf1_value, next_qf2_value)

        if next_barrier_value < safety_threshold:
            # Choose the action with the highest Q-value
            action = actor.predict(state)

        next_state, _, done, _ = env.step(action)
        state = next_state
        i += 1

    return i


def evaluate_constrain(
    actor: torch.nn.Module,
    qf1: torch.nn.Module,
    qf2: torch.nn.Module,
    env: gym.Env,
    safety_threshold: float = 0,
    num_rollouts: int = 10,
) -> pd.DataFrame:
    """Return pd.DataFrame of rollout data

    Each row is 1 episode
    """
    rows = []
    for episode_idx in range(num_rollouts):
        episode_length = evaluate_constrain_episode(
            actor, qf1, qf2, env, safety_threshold
        )
        rows.append(
            {
                "episode": episode_idx,
                "episode_length": episode_length,
            }
        )
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":

    from rl_cbf_extra.learning.td3_continuous_action import make_env, parse_args

    args = parse_args()
    _envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, "eval")]
    )
    actor = Actor(_envs)
    qf1 = QNetwork(_envs, bounded=args.bounded)
    qf2 = QNetwork(_envs, bounded=args.bounded)

    qf1.load_state_dict(torch.load("downloads/1qxv7iyl/qf1.pt"))
    qf2.load_state_dict(torch.load("downloads/1qxv7iyl/qf2.pt"))
    actor.load_state_dict(torch.load("downloads/1qxv7iyl/actor.pt"))

    env = gym.make(args.env_id)
    evaluate_constrain_episode(
        actor, qf1, qf2, env, safety_threshold=0.5 / (1 - 0.99), render=True
    )
    df = evaluate_constrain(actor, qf1, qf2, env, safety_threshold=0.5 / (1 - 0.99))
    print(df)
