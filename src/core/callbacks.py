from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from stable_baselines3.common.logger import Logger


class InfoRewardCallback:
    """Collect per-episode totals for numeric reward fields in environment infos."""

    def __init__(
        self,
        reward_keys: Sequence[str],
        logger: Logger | None = None,
        log_prefix: str = "eval/info_reward",
    ) -> None:
        if not reward_keys:
            raise ValueError("reward_keys must contain at least one info key")
        if len(set(reward_keys)) != len(reward_keys):
            raise ValueError("reward_keys must not contain duplicates")

        self.reward_keys = tuple(reward_keys)
        self.logger = logger
        self.log_prefix = log_prefix.rstrip("/")
        self.episode_rewards: dict[str, list[float]] = {key: [] for key in self.reward_keys}
        self._current_rewards: dict[int, dict[str, float]] = {}
        self._seen_rewards: dict[int, set[str]] = {}

    def __call__(self, locals_: dict[str, Any], _: dict[str, Any]) -> None:
        info = locals_["info"]
        env_index = locals_.get("i", 0)
        current_rewards = self._current_rewards.setdefault(env_index, {key: 0.0 for key in self.reward_keys})
        seen_rewards = self._seen_rewards.setdefault(env_index, set())

        for key in self.reward_keys:
            reward_components = info.get("rewards", {})
            if not isinstance(reward_components, Mapping):
                raise TypeError("info['rewards'] must be a mapping when present")
            if key in reward_components:
                value = reward_components[key]
            elif key in info:
                value = info[key]
            else:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Reward component {key!r} must be a real number, got {type(value).__name__}")
            current_rewards[key] += float(value)
            seen_rewards.add(key)

        if locals_["done"]:
            for key in seen_rewards:
                episode_reward = current_rewards[key]
                self.episode_rewards[key].append(episode_reward)
                if self.logger is not None:
                    self.logger.record_mean(f"{self.log_prefix}/{key}", episode_reward)
            del self._current_rewards[env_index]
            del self._seen_rewards[env_index]

    def reset(self) -> None:
        self.episode_rewards = {key: [] for key in self.reward_keys}
        self._current_rewards.clear()
        self._seen_rewards.clear()


class InfoRewardTrainingCallback(BaseCallback):
    """Collect per-episode reward components from training environment infos."""

    def __init__(self, reward_keys: Sequence[str]) -> None:
        super().__init__()
        self.reward_keys = tuple(reward_keys)
        self.info_reward_callback: InfoRewardCallback | None = None

    @property
    def episode_rewards(self) -> dict[str, list[float]]:
        if self.info_reward_callback is None:
            return {key: [] for key in self.reward_keys}
        return self.info_reward_callback.episode_rewards

    def _on_training_start(self) -> None:
        self.info_reward_callback = InfoRewardCallback(self.reward_keys)

    def _on_step(self) -> bool:
        if self.info_reward_callback is None:
            raise RuntimeError("InfoRewardTrainingCallback was not initialized")

        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for env_index, info in enumerate(infos):
            self.info_reward_callback(
                {"info": info, "i": env_index, "done": bool(dones[env_index])},
                {},
            )
        return True


class InfoRewardEvalCallback(EvalCallback):
    """EvalCallback that logs configured reward components from ``info["rewards"]``."""

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        reward_keys: Sequence[str],
        callback_on_new_best: BaseCallback | None = None,
        callback_after_eval: BaseCallback | None = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str | None = None,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        log_prefix: str = "eval/info_reward",
    ):
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.reward_keys = reward_keys
        self.log_prefix = log_prefix
        self.info_reward_callback: InfoRewardCallback | None = None

    def _init_callback(self) -> None:
        super()._init_callback()
        self.info_reward_callback = InfoRewardCallback(
            self.reward_keys,
            logger=self.logger,
            log_prefix=self.log_prefix,
        )

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        if self.info_reward_callback is None:
            raise RuntimeError("InfoRewardEvalCallback was not initialized")
        self.info_reward_callback(locals_, globals_)

    def _on_step(self) -> bool:
        should_evaluate = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if should_evaluate:
            if self.info_reward_callback is None:
                raise RuntimeError("InfoRewardEvalCallback was not initialized")
            self.info_reward_callback.reset()
        continue_training = super()._on_step()

        if should_evaluate and self.verbose >= 1:
            if self.info_reward_callback is None:
                raise RuntimeError("InfoRewardEvalCallback was not initialized")
            component_means = [
                f"{key}={sum(values) / len(values):.2f}"
                for key, values in self.info_reward_callback.episode_rewards.items()
                if values
            ]
            if component_means:
                print("Info reward means: " + ", ".join(component_means))

        return continue_training