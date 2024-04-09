# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Screw tasks with DClaw robots.

This is continuous rotation of an object to match a target velocity.
"""

from typing import Optional

import numpy as np

from robel.dclaw.turn import BaseDClawTurn
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable


class BaseDClawScrew(BaseDClawTurn):
    """Shared logic for DClaw screw tasks."""

    def __init__(self, success_threshold: float = 0.2, **kwargs):
        """Initializes the environment.

        Args:
            success_threshold: The difference threshold (in radians) of the
                object position and the goal position within which we consider
                as a sucesss.
        """
        super().__init__(success_threshold=success_threshold, **kwargs)

        # The target velocity is set during `_reset`.
        self._target_object_vel = 0
        self._desired_target_pos = 0

    def _reset(self):
        super()._reset()
        self._desired_target_pos = self._target_object_pos

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        # Update the target object goal.
        if not self._interactive:
            self._desired_target_pos += self._target_object_vel * self.dt
            self._set_target_object_pos(
                self._desired_target_pos, unbounded=True)
        super()._step(action)


@configurable(pickleable=True)
class DClawScrewFixed(BaseDClawScrew):
    """Rotates the object with a fixed initial position and velocity."""

    def _reset(self):
        # Start from the target and rotate at a constant velocity.
        self._initial_object_pos = 0
        self._set_target_object_pos(0)
        self._target_object_vel = 0.5
        super()._reset()


@configurable(pickleable=True)
class DClawScrewRandom(BaseDClawScrew):
    """Rotates the object with a random initial position and velocity."""

    def _reset(self):
        # Initial position is +/- 180 degrees.
        self._initial_object_pos = self.np_random.uniform(
            low=-np.pi, high=np.pi)
        self._set_target_object_pos(self._initial_object_pos)

        # Random target velocity.
        self._target_object_vel = self.np_random.uniform(low=-0.75, high=0.75)
        super()._reset()


@configurable(pickleable=True)
class DClawScrewRandomDynamics(DClawScrewRandom):
    """Rotates the object with a random initial position and velocity.

    The dynamics of the simulation are randomized each episode.
    """

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dclaw').qvel_indices.tolist() +
            self.robot.get_config('object').qvel_indices.tolist())

    def _reset(self):
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            damping_range=(0.005, 0.1),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(1, 3),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        self._randomizer.randomize_bodies(
            ['mount'],
            position_perturb_range=(-0.01, 0.01),
        )
        self._randomizer.randomize_geoms(
            ['mount'],
            color_range=(0.2, 0.9),
        )
        self._randomizer.randomize_geoms(
            parent_body_names=['valve'],
            color_range=(0.2, 0.9),
        )
        super()._reset()
