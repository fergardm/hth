from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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

__all__ = [
    "evaluate_policy_with_info_rewards",
    "load_agent_training_history",
    "plot_agent_training_curves",
    "plot_info_reward_episodes",
    "plot_training_trials",
]


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
    total_rewards: Sequence[float] | None = None,
    save_dir: str | Path | None = None,
    rolling_window: int = 1,
    best_episode: int | None = None,
) -> Figure:
    """Plot total reward and reward components in one vertically stacked figure.

    If ``rolling_window`` is greater than one, plot a rolling mean alongside the
    per-episode totals. If ``save_dir`` is provided, save one PNG there.
    """
    import matplotlib.pyplot as plt

    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")
    if best_episode is not None and best_episode < 1:
        raise ValueError("best_episode must be at least 1")
    if total_rewards is None and not info_rewards:
        raise ValueError("Provide total_rewards or at least one reward component to plot")

    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    plot_series: list[tuple[str, Sequence[float]]] = []
    if total_rewards is not None:
        plot_series.append(("Total reward", total_rewards))
    for key, values in info_rewards.items():
        plot_series.append((key, values))
    plot_series = [(label, values) for label, values in plot_series if len(values) > 0]
    if not plot_series:
        raise ValueError("There are no episode rewards to plot")

    figure, axes = plt.subplots(
        len(plot_series),
        1,
        figsize=(10, max(3, 2.8 * len(plot_series))),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes[:, 0]
    for axis, (label, values) in zip(flat_axes, plot_series):
        episodes = list(range(1, len(values) + 1))
        axis.plot(episodes, values, marker="o", alpha=0.35, label="Episode total")
        if rolling_window > 1:
            rolling_means = [
                sum(values[max(0, index - rolling_window + 1) : index + 1])
                / len(values[max(0, index - rolling_window + 1) : index + 1])
                for index in range(len(values))
            ]
            axis.plot(episodes, rolling_means, label=f"Rolling mean ({rolling_window} episodes)")
        if best_episode is not None:
            axis.axvline(best_episode, color="red", alpha=0.5, label="Best episode")
        axis.set(
            title=f"{label} per episode",
            ylabel="Reward",
        )
        axis.legend()
        axis.grid(True, alpha=0.3)

    flat_axes[-1].set_xlabel("Episode")
    figure.tight_layout()
    if output_dir is not None:
        figure.savefig(output_dir / "reward_curves.png")
    plt.show()
    return figure


def plot_training_trials(
    trials: Sequence[Mapping[str, Any]],
    save_path: str | Path | None = None,
    rolling_window: int = 10,
) -> Figure:
    """Compare total training rewards and reward components across trials.

    Each trial mapping must contain ``trial_number``, ``params``,
    ``total_rewards``, ``info_rewards``, and may contain ``best_episode``.
    """
    import matplotlib.pyplot as plt

    if not trials:
        raise ValueError("trials must contain at least one completed training trial")
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    reward_keys = list(dict.fromkeys(
        key
        for trial in trials
        for key in trial["info_rewards"]
    ))
    series = [("Total reward", "total_rewards")] + [
        (key, key) for key in reward_keys
    ]
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(11, max(3, 2.8 * len(series))),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes[:, 0]
    colors = plt.get_cmap("tab10")

    for trial_index, trial in enumerate(trials):
        trial_number = trial["trial_number"]
        params = trial["params"]
        config_label = ", ".join(f"{key}={value}" for key, value in params.items())
        label = f"Trial {trial_number} ({config_label})"
        color = colors(trial_index % 10)

        for axis, (series_label, data_key) in zip(flat_axes, series):
            if data_key == "total_rewards":
                values = trial["total_rewards"]
            else:
                values = trial["info_rewards"].get(data_key, ())
            if len(values) == 0:
                continue

            episodes = list(range(1, len(values) + 1))
            axis.plot(episodes, values, color=color, alpha=0.2, linewidth=0.8, label="_nolegend_")
            if rolling_window > 1:
                smoothed = [
                    sum(values[max(0, index - rolling_window + 1) : index + 1])
                    / len(values[max(0, index - rolling_window + 1) : index + 1])
                    for index in range(len(values))
                ]
                axis.plot(
                    episodes,
                    smoothed,
                    color=color,
                    label=label,
                )
            else:
                axis.plot(episodes, values, color=color, alpha=0.8, label=label)

            best_episode = trial.get("best_episode")
            if best_episode is not None:
                axis.axvline(
                    best_episode,
                    color="red",
                    alpha=0.5,
                    linestyle="--",
                    label=f"Trial {trial_number} best episode" if axis is flat_axes[0] else None,
                )

    for axis, (series_label, _) in zip(flat_axes, series):
        axis.set(title=f"{series_label} per episode", ylabel="Reward")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    flat_axes[-1].set_xlabel("Training episode")

    figure.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
    plt.show()
    return figure


def load_agent_training_history(path: str | Path) -> dict[str, Any]:
    """Load a history archive saved by the PPO training notebook."""
    import numpy as np

    history_path = Path(path)
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Training history not found: {history_path}. "
            "Rerun training for this agent to create training_history.npz."
        )

    with np.load(history_path, allow_pickle=False) as archive:
        if "total_rewards" not in archive:
            raise ValueError(f"{history_path} does not contain total_rewards")
        info_rewards = {
            key.removeprefix("info_reward__"): archive[key].copy()
            for key in archive.files
            if key.startswith("info_reward__")
        }
        best_episode = None
        if "best_episode" in archive:
            value = int(archive["best_episode"].item())
            best_episode = value if value > 0 else None
        configuration = (
            str(archive["configuration"].item())
            if "configuration" in archive
            else ""
        )
        return {
            "total_rewards": archive["total_rewards"].copy(),
            "info_rewards": info_rewards,
            "best_episode": best_episode,
            "configuration": configuration,
        }


def plot_agent_training_curves(
    agents: Mapping[str, Mapping[str, Any]],
    save_path: str | Path | None = None,
    rolling_window: int = 10,
) -> Figure:
    """Compare training curves for separately trained agents or wrapper setups."""
    import matplotlib.pyplot as plt

    if not agents:
        raise ValueError("agents must contain at least one agent history")
    if rolling_window < 1:
        raise ValueError("rolling_window must be at least 1")

    reward_keys = list(dict.fromkeys(
        key
        for history in agents.values()
        for key in history.get("info_rewards", {})
    ))
    series = [("Total reward", "total_rewards")] + [
        (key, key) for key in reward_keys
    ]
    figure, axes = plt.subplots(
        len(series),
        1,
        figsize=(11, max(3, 2.8 * len(series))),
        sharex=True,
        squeeze=False,
    )
    flat_axes = axes[:, 0]
    color_map = plt.get_cmap("tab10")

    for agent_index, (agent_name, history) in enumerate(agents.items()):
        color = color_map(agent_index % 10)
        label = agent_name

        for axis, (_, data_key) in zip(flat_axes, series):
            values = (
                history.get("total_rewards", ())
                if data_key == "total_rewards"
                else history.get("info_rewards", {}).get(data_key, ())
            )
            if len(values) == 0:
                continue

            episodes = list(range(1, len(values) + 1))
            axis.plot(
                episodes,
                values,
                color=color,
                alpha=0.2,
                linewidth=0.8,
                label="_nolegend_",
            )
            if rolling_window > 1:
                smoothed = [
                    sum(values[max(0, index - rolling_window + 1) : index + 1])
                    / len(values[max(0, index - rolling_window + 1) : index + 1])
                    for index in range(len(values))
                ]
                axis.plot(episodes, smoothed, color=color, label=label)
            else:
                axis.plot(episodes, values, color=color, alpha=0.8, label=label)

            best_episode = history.get("best_episode")
            if best_episode is not None:
                axis.axvline(
                    best_episode,
                    color=color,
                    alpha=0.5,
                    linestyle="--",
                    label=f"{agent_name} best episode" if axis is flat_axes[0] else None,
                )

    for axis, (series_name, _) in zip(flat_axes, series):
        axis.set(title=f"{series_name} per episode", ylabel="Reward")
        axis.grid(True, alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    flat_axes[-1].set_xlabel("Training episode")

    figure.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path)
    plt.show()
    return figure