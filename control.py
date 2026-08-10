"""
Baseline longitudinal controller and noise-injection wrapper.

CAVEAT: "the paper's rule-compliant longitudinal controller (Section V-C)"
is not reproduced here — that section's exact control law wasn't available.
BaselineController below is a standard gap-following (constant-time-headway)
controller with a speed-limit ceiling, provided as a functional stand-in so
category (a)/(b) collection can run end-to-end. Replace `BaselineController.step`
with the paper's actual control law before treating collected data as
representative of "the rule-compliant policy" from the paper.
"""

import random
from collections import deque

import carla


class BaselineController:
    """Simple constant-time-headway gap follower with a speed-limit ceiling.
    Rule-compliant by construction in the sense that it brakes proportionally
    to closing gap and never exceeds the target speed — but this is NOT
    verified against the paper's formal rulebook definition."""

    def __init__(self, target_speed_kmh: float = 40.0, time_headway: float = 1.5, max_brake: float = 1.0):
        self.target_speed = target_speed_kmh / 3.6  # m/s
        self.time_headway = time_headway
        self.max_brake = max_brake

    def step(self, ego_speed: float, lead_distance, lead_relative_speed) -> carla.VehicleControl:
        control = carla.VehicleControl()

        desired_gap = max(2.0, self.time_headway * ego_speed)
        if lead_distance is not None and lead_distance < desired_gap * 2.0:
            gap_error = lead_distance - desired_gap
            # closing_speed > 0 means lead is receding in our sign convention (see rulebook.py);
            # brake harder when gap_error is negative and closing fast.
            brake_signal = max(0.0, -gap_error / desired_gap - 0.3 * min(0.0, lead_relative_speed or 0.0))
            control.brake = min(self.max_brake, brake_signal)
            control.throttle = 0.0 if control.brake > 0.05 else 0.3
        else:
            speed_error = self.target_speed - ego_speed
            control.throttle = max(0.0, min(0.8, 0.3 + 0.1 * speed_error))
            control.brake = 0.0 if speed_error > -0.5 else min(self.max_brake, -speed_error * 0.2)

        control.steer = 0.0  # steering left to CARLA's autopilot / lane-follow separately
        return control


class NoiseInjector:
    """Wraps a control signal with the perturbations described for category
    (b): Gaussian noise on accel/steer, randomized brake dropout, and
    reaction-time delay via a ring buffer of past commands."""

    def __init__(self, accel_noise_std: float = 0.05, steer_noise_std: float = 0.03,
                 brake_dropout_prob: float = 0.15, max_delay_ticks: int = 6, seed: int = 0):
        self.accel_noise_std = accel_noise_std
        self.steer_noise_std = steer_noise_std
        self.brake_dropout_prob = brake_dropout_prob
        self.rng = random.Random(seed)
        self.delay_ticks = self.rng.randint(0, max_delay_ticks)
        self.buffer = deque(maxlen=max(1, self.delay_ticks + 1))
        self._episode_drops_brake = self.rng.random() < brake_dropout_prob

    def apply(self, control: carla.VehicleControl) -> carla.VehicleControl:
        noisy = carla.VehicleControl()
        noisy.throttle = min(1.0, max(0.0, control.throttle + self.rng.gauss(0, self.accel_noise_std)))
        noisy.brake = min(1.0, max(0.0, control.brake + self.rng.gauss(0, self.accel_noise_std)))
        noisy.steer = min(1.0, max(-1.0, control.steer + self.rng.gauss(0, self.steer_noise_std)))

        if self._episode_drops_brake:
            noisy.brake = 0.0  # simulate missed/delayed brake response for this whole episode

        # reaction-time delay: push current command, pop delayed one
        self.buffer.append(noisy)
        if len(self.buffer) < self.buffer.maxlen:
            return noisy  # not enough history yet, pass through
        return self.buffer[0]
