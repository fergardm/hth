from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym

from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecEnv

from .callbacks import InfoRewardCallback


def evaluate_policy_with_info_rewards(
    model: type_aliases.PolicyPredictor,
    env: gym.Env | VecEnv,
    reward_keys: Sequence[str],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    logger: Logger | None = None,
    log_prefix: str = "eval/info_reward",
) -> tuple[tuple[float, float] | tuple[list[float], list[int]], dict[str, list[float]]]:
    """Evaluate a policy and collect per-episode totals for selected info fields.

    Info values are summed over each episode. Metrics that are absent for an entire
    episode are omitted from that metric's result list.
    """
    info_reward_callback = InfoRewardCallback(reward_keys, logger=logger, log_prefix=log_prefix)

    def combined_callback(locals_: dict[str, Any], _globals: dict[str, Any]) -> None:
        info_reward_callback(locals_, _globals)
        if callback is not None:
            callback(locals_, _globals)

    evaluation_result = sb3_evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=deterministic,
        render=render,
        callback=combined_callback,
        reward_threshold=reward_threshold,
        return_episode_rewards=return_episode_rewards,
        warn=warn,
    )
    return evaluation_result, info_reward_callback.episode_rewards