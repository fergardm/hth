# hth
Hackway to Hell repo

## Evaluating info rewards

Use `InfoRewardEvalCallback` to collect reward components placed in each environment's `info["rewards"]` dictionary during periodic Stable-Baselines3 evaluation:

```python
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from src.core.callbacks import InfoRewardEvalCallback

info_reward_eval_callback = InfoRewardEvalCallback(
    eval_env,
    reward_keys=("collision_reward", "right_lane_reward", "high_speed_reward", "on_road_reward"),
    n_eval_episodes=10,
    eval_freq=40,
    best_model_save_path="./logs/",
    log_path="./logs/",
)

callbacks = CallbackList([info_reward_eval_callback, CheckpointCallback(save_freq=40, save_path="./logs/")])
```

This replaces `EvalCallback` in the callback list; do not add `InfoRewardCallback` directly to `CallbackList`, as it is the per-step collector used internally during evaluation. Completed-episode means are recorded under `eval/info_reward/<key>`. Values in `info["rewards"]` are summed across each episode; direct `info` values are also supported. The last evaluation's per-episode component totals are available from `info_reward_eval_callback.info_reward_callback.episode_rewards`.

To plot the total return on top and each reward component beneath it after a standalone evaluation:

```python
from src.core.evaluation import evaluate_policy_with_info_rewards, plot_info_reward_episodes

(episode_returns, episode_lengths), info_rewards = evaluate_policy_with_info_rewards(
    agent,
    eval_env,
    reward_keys=("collision_reward", "right_lane_reward", "high_speed_reward", "on_road_reward"),
    return_episode_rewards=True,
)
figure = plot_info_reward_episodes(
    info_rewards,
    total_rewards=episode_returns,
    save_dir="./logs/reward_plots",
)
```

The helper saves a single `reward_curves.png`. Each panel shows the **episode total** against episode number, not per-timestep rewards.

## Keeping and plotting the best training episode

To track reward components from training (rather than evaluation), use `BestEpisodeCallback`. It collects per-episode component totals and saves the policy checkpoint when a completed episode has a higher combined reward than all previous episodes:

```python
from src.core.callbacks import BestEpisodeCallback
from src.core.evaluation import plot_info_reward_episodes

best_episode_callback = BestEpisodeCallback(
    ("collision_reward", "right_lane_reward", "high_speed_reward", "on_road_reward"),
    best_model_save_path="./logs/best_episode_model.zip",
)

# Include best_episode_callback in the CallbackList passed to model.learn().
plot_info_reward_episodes(
    best_episode_callback.episode_rewards,
    total_rewards=best_episode_callback.episode_returns,
    save_dir="./logs/training_reward_plots",
    rolling_window=10,
    best_episode=best_episode_callback.best_episode,
)
```

The callback exposes `best_reward`, `best_episode`, and `best_episode_components`. The plot marks the best episode with a red vertical line. Training curves are available only for episodes observed while this callback was enabled, so rerun training to collect them.

## Comparing training trials

The PPO notebook can retain each trial's training returns, component totals, and configuration, then compare them in one stacked figure:

```python
from src.core.evaluation import plot_training_trials

plot_training_trials(
    training_trial_results,
    save_path="./logs/training_trial_comparison.png",
    rolling_window=10,
)
```

The top panel compares total training return, followed by one panel per reward component. Each trial's rolling-mean curve is labeled with its configuration; red dashed markers show that trial's highest-reward episode. Set `N_COMBINACIONES` to the desired number of trials in the notebook.
