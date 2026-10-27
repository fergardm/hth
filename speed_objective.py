"""Training-only speed objective; evaluation retains original highway rewards."""

from math import isfinite


def speed_reward(components, crashed, collision_penalty=2.0, right_lane_weight=0.1):
    """Use highway's forward-speed component, which is 0 below 20 and 1 at 30 m/s.

    No positive survival offset: a slow agent cannot collect a large return just
    by staying alive. No bonus for issuing FASTER: reward actual achieved speed.
    Right-lane reward is gated by speed so lane occupancy alone cannot replace
    progress. This changes the training objective and must be compared with the
    unmodified evaluation objective. Coefficients are experimental defaults.
    """
    if not isfinite(collision_penalty) or collision_penalty <= 0:
        raise ValueError("collision_penalty must be positive and finite")
    if not isfinite(right_lane_weight) or right_lane_weight < 0:
        raise ValueError("right_lane_weight must be nonnegative and finite")
    if crashed:
        return -float(collision_penalty)
    speed = float(components["high_speed_reward"])
    lane = float(components["right_lane_reward"])
    on_road = float(components["on_road_reward"])
    if not all(isfinite(value) for value in (speed, lane, on_road)):
        raise ValueError("Reward components must be finite")
    speed, lane = min(max(speed, 0.0), 1.0), min(max(lane, 0.0), 1.0)
    return speed * (1.0 + right_lane_weight * lane) * min(max(on_road, 0.0), 1.0)


def apply_speed_objective(env, *, collision_penalty=2.0, right_lane_weight=0.1):
    """Construct a Gymnasium wrapper without requiring Gymnasium at module import."""
    import gymnasium as gym

    # Fail early on invalid arguments, rather than during a training episode.
    speed_reward({"high_speed_reward": 0, "right_lane_reward": 0,
                  "on_road_reward": 1}, False, collision_penalty, right_lane_weight)

    class SpeedObjectiveReward(gym.Wrapper):
        def step(self, action):
            obs, original_reward, terminated, truncated, info = self.env.step(action)
            reward = speed_reward(info["rewards"], info["crashed"],
                                  collision_penalty, right_lane_weight)
            info = dict(info)
            info["original_reward"] = float(original_reward)
            return obs, reward, terminated, truncated, info

    return SpeedObjectiveReward(env)