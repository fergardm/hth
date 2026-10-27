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

To plot a separate per-episode trend for each reward component after a standalone evaluation:

```python
from src.core.evaluation import evaluate_policy_with_info_rewards, plot_info_reward_episodes

(episode_returns, episode_lengths), info_rewards = evaluate_policy_with_info_rewards(
    agent,
    eval_env,
    reward_keys=("collision_reward", "right_lane_reward", "high_speed_reward", "on_road_reward"),
    return_episode_rewards=True,
)
figures = plot_info_reward_episodes(info_rewards, save_dir="./logs/reward_plots")
```

Each plot shows the **total component reward for an episode** against episode number. These are episode totals, not per-timestep rewards.

## Plotting training info rewards

To track reward components from training (rather than evaluation), add `InfoRewardTrainingCallback` to the training callbacks. After training, its `episode_rewards` contains one total per completed training episode and can be plotted directly:

```python
from src.core.callbacks import InfoRewardTrainingCallback
from src.core.evaluation import plot_info_reward_episodes

info_reward_training_callback = InfoRewardTrainingCallback(
    ("collision_reward", "right_lane_reward", "high_speed_reward", "on_road_reward")
)

# Include info_reward_training_callback in the CallbackList passed to model.learn().
plot_info_reward_episodes(
    info_reward_training_callback.episode_rewards,
    save_dir="./logs/training_reward_plots",
    rolling_window=10,
)
```

The training curves are available only for episodes observed while this callback was enabled, so rerun training to collect them.
