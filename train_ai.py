"""Train and select a highway PPO agent with observed lane-safety features.

Run from the repository root: python workspace/train_highway.py
Use run_wrapper_smoke.py first to validate the installed runtime.
"""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
from statistics import mean, pstdev
import time


def selection_key(metrics, collision_limit):
    """Reward first within the empirical safety limit; collisions first outside it."""
    rate, reward = metrics["collision_rate"], metrics["mean_reward"]
    if rate <= collision_limit:
        return (1, reward, -rate)
    return (0, -rate, reward)


def summarize(episodes):
    rewards = [r["reward"] for r in episodes]
    return {
        "mean_reward": mean(rewards), "reward_std": pstdev(rewards),
        "collision_rate": mean(int(r["collision"]) for r in episodes),
        "mean_speed_m_s": mean(r["mean_speed_m_s"] for r in episodes),
        "mean_right_lane_fraction": mean(r["right_lane_fraction"] for r in episodes),
        "mean_steps": mean(r["steps"] for r in episodes), "episodes": episodes,
    }


def evaluate(env, agent, seeds):
    """Original, undiscounted rewards; agent=None is the constant-idle baseline."""
    episodes = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        done = False
        reward_sum, speeds, right_lane = 0.0, [], []
        collided = False
        while not done:
            action = 1 if agent is None else agent.predict(observation, deterministic=True)[0]
            observation, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            speeds.append(float(info["speed"]))
            # The wrapper's final feature is current normalized lane position.
            right_lane.append(float(observation[-1]) >= 1.0 - 1e-6)
            collided = collided or bool(info["crashed"])
            done = terminated or truncated
        episodes.append({"seed": seed, "reward": reward_sum,
                         "collision": collided, "steps": len(speeds),
                         "mean_speed_m_s": mean(speeds),
                         "right_lane_fraction": mean(int(v) for v in right_lane)})
    return summarize(episodes)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--eval-freq", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-collision-rate", type=float, default=0.10,
                        help="Empirical validation constraint, not a safety guarantee (default .10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--train-env", choices=["highway-v0", "highway-fast-v0"],
                        default="highway-v0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.timesteps < 12 or args.eval_freq < 1 or args.eval_episodes < 1
            or args.threads < 1 or args.seed < 0 or not 0 <= args.max_collision_rate <= 1):
        parser.error("Require timesteps >= 512, positive counts, seed >= 0 and collision rate in [0, 1]")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import gymnasium as gym
        import highway_env  # Registers environment IDs.
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from wrapper import aplicar_wrappers
    except ImportError as error:
        raise SystemExit(f"Missing runtime dependency: {error}. Install requirements.txt first.") from error

    torch.set_num_threads(args.threads)
    output = args.output or Path(__file__).resolve().parent / "agentes" / (
        "PPO-highway-safety-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output"] = str(output.resolve())
    validation_seeds = list(range(10_000, 10_000 + args.eval_episodes))
    test_seeds = list(range(200, 220))
    write_json(output / "run_config.json", {
        **config, "validation_seeds": validation_seeds, "test_seeds": test_seeds,
        "versions": {name: metadata.version(name) for name in
                     ("torch", "numpy", "gymnasium", "highway-env", "stable-baselines3")},
        "selection": "Maximize original reward within collision constraint; otherwise minimize collisions",
        "reward_shaping": False, "algorithm": "PPO",
        "ppo": {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 64,
                "n_epochs": 10, "gamma": .99, "gae_lambda": .95,
                "clip_range": .2, "ent_coef": .01, "target_kl": .03,
                "net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    })
    shutil.copyfile(Path(__file__).with_name("wrapper.py"), output / "wrappers.py")
    (output / "algoritmo.txt").write_text("PPO\n", encoding="utf-8")
    train_env = validation_env = None
    started = time.perf_counter()

    class SafetyValidation(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best_key = None
            self.best_metrics = None
            self.last_evaluation = None

        def assess(self):
            metrics = evaluate(validation_env, self.model, validation_seeds)
            key = selection_key(metrics, args.max_collision_rate)
            self.last_evaluation = self.num_timesteps
            record = {"timesteps": self.num_timesteps, **metrics}
            if self.best_key is None or key > self.best_key:
                self.best_key, self.best_metrics = key, record
                self.model.save(output / "best_model.zip")
                write_json(output / "best_validation.json", record)
            with (output / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"Validation step {self.num_timesteps}: reward={metrics['mean_reward']:.2f}, "
                  f"collisions={metrics['collision_rate']:.1%}, "
                  f"speed={metrics['mean_speed_m_s']:.2f} m/s", flush=True)

        def _on_step(self):
            return True

        def _on_rollout_start(self):
            # PPO updates AFTER collecting a rollout. Evaluate at the next
            # rollout start so saved checkpoints include the completed update.
            if self.num_timesteps - (self.last_evaluation or 0) >= args.eval_freq:
                self.assess()

        def _on_training_end(self):
            # Includes the last PPO update, even when rollout length overshoots
            # the requested timestep budget.
            self.assess()

    try:
        validation_env = aplicar_wrappers(gym.make("highway-v0"))
        check_env(validation_env, warn=True)
        baseline = evaluate(validation_env, None, validation_seeds)
        write_json(output / "baseline_validation.json", baseline)
        train_env = Monitor(aplicar_wrappers(gym.make(args.train_env)),
                            filename=str(output / "train"))
        model = PPO(
            "MlpPolicy", train_env, seed=args.seed, device=args.device, verbose=1,
            learning_rate=3e-4, n_steps=512, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
            target_kl=0.03,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        )
        validation = SafetyValidation()
        model.learn(total_timesteps=args.timesteps, callback=validation, log_interval=1)
        model.save(output / "final_model.zip")
        # Selection has finished; test seeds never participate in checkpoint selection.
        selected = PPO.load(output / "best_model.zip", env=validation_env, device=args.device)
        test_baseline = evaluate(validation_env, None, test_seeds)
        test = evaluate(validation_env, selected, test_seeds)
        write_json(output / "test_results.json", {
            "status": "completed", "elapsed_seconds": time.perf_counter() - started,
            "timesteps": model.num_timesteps, "baseline": test_baseline, "selected_model": test,
            "best_validation_timesteps": validation.best_metrics["timesteps"],
            "validation_collision_constraint_met": (
                validation.best_metrics["collision_rate"] <= args.max_collision_rate),
            "test_collision_constraint_met": test["collision_rate"] <= args.max_collision_rate,
            "test_reward_improves_over_idle": test["mean_reward"] > test_baseline["mean_reward"],
            "note": "Empirical results on 20 test episodes; not a collision-probability guarantee.",
        })
        print(f"Saved selected model and results to {output.resolve()}", flush=True)
        print(f"Test reward: {test['mean_reward']:.2f}; idle: {test_baseline['mean_reward']:.2f}; "
              f"collisions: {test['collision_rate']:.1%}")
    finally:
        if train_env is not None:
            train_env.close()
        if validation_env is not None:
            validation_env.close()


if __name__ == "__main__":
    main()
