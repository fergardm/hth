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

This replaces `EvalCallback` in the callback list; do not add `InfoRewardCallback` directly to `CallbackList`, as it is the per-step collector used internally during evaluation. Completed-episode means are recorded under `eval/info_reward/<key>`. Values in `info["rewards"]` are summed across each episode; direct `info` values are also supported. The last evaluation's qqqper-episode component totals are available from `info_reward_eval_callbackq.info_reward_callback.episode_rewards`.
