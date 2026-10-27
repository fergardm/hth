"""Lane safety features for highway-v0 and highway-fast-v0.

In both train.ipynb and test.ipynb, replace the local wrapper hook with::

    from wrapper import aplicar_wrappers

Retrain the policy after enabling this wrapper: the observation becomes a flat
vector (59 values with the default 5-by-5 kinematics observation). For submission,
copy this file as wrappers.py, as required by the hackathon README.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from highway_env.envs.common.observation import KinematicObservation
from highway_env.road.lane import AbstractLane


class LaneSafetyObservation(gym.ObservationWrapper):
    """Append fixed-scale safety summaries to the unchanged, flattened input.

Uses only observed vehicle rows plus static road/observation configuration; it
never reads simulator traffic or the ego vehicle's hidden state. Supports the
straight, equal-width lanes of the two highway environments, not curved roads.

For LEFT, CURRENT, RIGHT, append these 11 features in order:
    exists,
    front_seen, front_reliable, front_gap, front_closing_speed, front_ttc,
    rear_seen, rear_reliable, rear_gap, rear_closing_speed, rear_ttc.
Finally append lane_position (0 = leftmost, 1 = rightmost).

Gaps use an assumed common vehicle length (5 m by default); TTC assumes constant
longitudinal velocities and ignores lateral motion. Positive closing speed means
the gap is shrinking. Gap, signed closing speed, and TTC are divided by their
respective caps and clipped to [0, 1], [-1, 1], and [0, 1]. These are estimates,
not guarantees of safety.

Missing neighbors have seen=reliable=0 and numeric features=0: they MUST NOT be
interpreted as clear road. The default observation limits vehicle count and rear
visibility. Clipped lateral positions cannot be assigned to a lane and are
excluded; clipped longitudinal positions/velocities set reliable=0 and numeric
features=0. An unclipped observed neighbor still does not establish full coverage.
No learned statistics or additional model files are needed.
"""

    LANE_FEATURES = (
        "exists", "front_seen", "front_reliable", "front_gap",
        "front_closing_speed", "front_ttc", "rear_seen", "rear_reliable",
        "rear_gap", "rear_closing_speed", "rear_ttc",
    )

    def __init__(self, env, *, gap_cap=100.0, closing_speed_cap=30.0,
                 ttc_cap=10.0, vehicle_length=5.0):
        super().__init__(env)
        if env.unwrapped.spec is None or env.unwrapped.spec.id not in (
            "highway-v0", "highway-fast-v0"
        ):
            raise ValueError("LaneSafetyObservation supports only highway environments")
        self.kinematics = env.unwrapped.observation_type
        if not isinstance(self.kinematics, KinematicObservation):
            raise ValueError("A Kinematics observation is required")
        if not isinstance(env.observation_space, spaces.Box):
            raise ValueError("A matrix Box observation is required")
        self.features = list(self.kinematics.features)
        required = ("presence", "x", "y", "vx", "vy")
        if not all(name in self.features for name in required):
            raise ValueError(f"Kinematics must include {required}")
        if env.observation_space.shape != (
            self.kinematics.vehicles_count, len(self.features)
        ):
            raise ValueError("Apply LaneSafetyObservation before other observation wrappers")
        for name, value in (("gap_cap", gap_cap),
                            ("closing_speed_cap", closing_speed_cap),
                            ("ttc_cap", ttc_cap), ("vehicle_length", vehicle_length)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, float(value))
        self.lanes_count = int(env.unwrapped.config["lanes_count"])
        self.lane_width = float(AbstractLane.DEFAULT_WIDTH)
        self.input_shape = env.observation_space.shape
        self.feature_names = tuple(
            f"{lane}_{feature}"
            for lane in ("left", "current", "right")
            for feature in self.LANE_FEATURES
        ) + ("lane_position",)
        extra_low = np.zeros(len(self.feature_names), dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if name.endswith("closing_speed"):
                extra_low[i] = -1.0
        self.observation_space = spaces.Box(
            low=np.concatenate((env.observation_space.low.ravel(), extra_low)),
            high=np.concatenate((env.observation_space.high.ravel(),
                                 np.ones(len(self.feature_names), dtype=np.float32))),
            dtype=np.float32,
        )

    def _decode(self, observation):
        """Undo fixed normalization; flag clipped values that lost information."""
        raw = np.array(observation, dtype=np.float64, copy=True)
        clipped = np.zeros(raw.shape, dtype=bool)
        if self.kinematics.normalize:
            # highway-env initializes default ranges on the first reset, not
            # necessarily when this wrapper is constructed.
            ranges = self.kinematics.features_range
            if not ranges:
                raise ValueError("Observation normalization ranges are not initialized")
            for name, (low, high) in ranges.items():
                if name not in self.features:
                    continue
                j = self.features.index(name)
                if self.kinematics.clip:
                    clipped[:, j] = np.abs(raw[:, j]) >= 1.0 - 1e-6
                raw[:, j] = low + (raw[:, j] + 1.0) * (high - low) / 2.0
        return raw, clipped

    def observation(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != self.input_shape or not np.isfinite(observation).all():
            raise ValueError("Expected a finite kinematics matrix with the original shape")
        raw, clipped = self._decode(observation)
        p, x, y, vx = [self.features.index(k) for k in ("presence", "x", "y", "vx")]
        if raw[0, p] <= 0 or clipped[0, y]:
            raise ValueError("An observable ego vehicle and lateral position are required")
        ego_lane = int(np.clip(np.floor(raw[0, y] / self.lane_width + 0.5),
                               0, self.lanes_count - 1))
        relative = raw.copy()
        if self.kinematics.absolute:
            relative[1:, [x, y, vx]] -= raw[0, [x, y, vx]]
        # In relative mode only neighbor rows are relative; row zero is absolute.
        lane_ids = np.floor((raw[0, y] + relative[:, y]) / self.lane_width + 0.5)
        available = (raw[:, p] > 0) & ~clipped[:, y]
        available[0] = False
        summaries = []
        for lane in (ego_lane - 1, ego_lane, ego_lane + 1):
            block = np.zeros(11, dtype=np.float32)
            if 0 <= lane < self.lanes_count:
                block[0] = 1.0
                in_lane = available & (lane_ids == lane)
                for front, offset in ((True, 1), (False, 6)):
                    candidates = np.flatnonzero(in_lane & (
                        (relative[:, x] >= 0) if front else (relative[:, x] < 0)
                    ))
                    if candidates.size:
                        index = candidates[np.argmin(np.abs(relative[candidates, x]))]
                        block[offset] = 1.0
                        reliable = not (clipped[index, x] or clipped[index, vx])
                        if self.kinematics.absolute:
                            reliable = reliable and not (clipped[0, x] or clipped[0, vx])
                        if reliable:
                            gap = max(0.0, abs(relative[index, x]) - self.vehicle_length)
                            closing = -relative[index, vx] if front else relative[index, vx]
                            ttc = gap / closing if closing > 1e-6 else self.ttc_cap
                            if gap == 0:
                                ttc = 0.0
                            block[offset + 1:offset + 5] = (
                                1.0, min(gap / self.gap_cap, 1.0),
                                np.clip(closing / self.closing_speed_cap, -1.0, 1.0),
                                min(ttc / self.ttc_cap, 1.0),
                            )
            summaries.extend(block)
        summaries.append(ego_lane / max(self.lanes_count - 1, 1))
        return np.concatenate((observation.ravel(),
                               np.asarray(summaries, dtype=np.float32)))

class PremiarVelocidad(gym.RewardWrapper):
    def reward(self, recompensa):
        coche = self.unwrapped.vehicle

        if coche.crashed or not coche.on_road:
            return recompensa

        delantero, _ = self.unwrapped.road.neighbour_vehicles(
            coche, coche.lane_index
        )

        espacio = float("inf")
        if delantero is not None:
            espacio = (
                coche.lane_distance_to(delantero)
                - (coche.LENGTH + delantero.LENGTH) / 2
            )

        # Premiar velocidad si hay suficiente espacio delante.
        if espacio > max(15.0, 2.0 * coche.speed):
            bonus = min(1.0, max(0.0, (coche.speed - 20.0) / 10.0))
            recompensa += 0.1 * bonus

        return recompensa

class PremiarVelocidadAgresivo(gym.RewardWrapper):
    def reward(self, recompensa):
        coche = self.unwrapped.vehicle

        if coche.crashed or not coche.on_road:
            return recompensa

        # Premio creciente entre 20 y 30 m/s.
        velocidad = min(1.0, max(0.0, (coche.speed - 20.0) / 10.0))

        return recompensa + 0.2 * velocidad

class AnticiparColision(gym.RewardWrapper):
    def reward(self, recompensa):
        coche = self.unwrapped.vehicle

        if coche.crashed:
            return recompensa - 1.0

        delantero, _ = self.unwrapped.road.neighbour_vehicles(
            coche, coche.lane_index
        )

        if delantero is not None:
            hueco = (
                coche.lane_distance_to(delantero)
                - (coche.LENGTH + delantero.LENGTH) / 2
            )
            acercamiento = coche.velocity[0] - delantero.velocity[0]

            if acercamiento > 0:
                ttc = max(0.0, hueco) / acercamiento
                if ttc < 3.0:
                    recompensa -= 0.2 * (1.0 - ttc / 3.0)

        return recompensa

class EvitarAtasco(gym.RewardWrapper):
    def reward(self, recompensa):
        coche = self.unwrapped.vehicle
        carretera = self.unwrapped.road

        if coche.crashed or not coche.on_road:
            return recompensa

        delantero, _ = carretera.neighbour_vehicles(coche, coche.lane_index)

        # ¿Vamos lentos y tenemos un coche cerca delante?
        bloqueado = (
            delantero is not None
            and coche.speed < 25.0
            and coche.lane_distance_to(delantero) < 35.0
        )

        if bloqueado:
            for carril in carretera.network.side_lanes(coche.lane_index):
                delante, detras = carretera.neighbour_vehicles(coche, carril)

                libre_delante = (
                    delante is None
                    or coche.lane_distance_to(delante) > 50.0
                )
                libre_detras = (
                    detras is None
                    or coche.lane_distance_to(detras) < -30.0
                )

                if libre_delante and libre_detras:
                    recompensa -= 0.05
                    break

        return recompensa

def aplicar_wrappers(env):
    """Shared train/test hook; preserves actions, rewards, termination and info."""
    return LaneSafetyObservation(env)
