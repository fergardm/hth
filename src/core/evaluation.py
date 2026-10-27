from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym

from stable_baselines3.common import type_aliases
from stable_baselines3.common.evaluation import evaluate_policy as sb3_evaluate_policy
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import VecEnv

from .callbacks import InfoRewardCallback

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["evaluate_policy_with_info_rewards", "plot_info_reward_episodes"]


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


def plot_info_reward_episodes(
    info_rewards: dict[str, Sequence[float]],
    save_dir: str | Path | None = None,
    rolling_window: int = 1,
) -> dict[str, Figure]:
    """Plot each reward component's total by episode in its own figure.

    If ``rolling_window`` is greater than one, plot a rolling mean alongside the
    per-episode totals. If ``save_dir`` is provided, save each plot there as a PNG.
    Figures are also displayed and returned by reward key for further customization.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Figure] = {}
    for key, values in info_rewards.items():
        if len(values) == 0:
            continue

        figure, axis = plt.subplots()
        episodes = list(range(1, len(values) + 1))
        axis.plot(episodes, values, marker="o", alpha=0.35, label="Episode total")
        if rolling_window > 1:
            rolling_means = [
                sum(values[max(0, index - rolling_window + 1) : index + 1])
                / len(values[max(0, index - rolling_window + 1) : index + 1])
                for index in range(len(values))
            ]
            axis.plot(episodes, rolling_means, label=f"Rolling mean ({rolling_window} episodes)")
        axis.set(
            title=f"{key} per episode",
            xlabel="Episode",
            ylabel="Episode total reward",
        )
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        if output_dir is not None:
            figure.savefig(output_dir / f"{key.replace('/', '_')}.png")
        figures[key] = figure

    if figures:
        plt.show()
    return figures