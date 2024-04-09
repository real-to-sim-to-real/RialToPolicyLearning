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

"""Configuration for a tracker component group."""

from typing import Iterable, Optional

import numpy as np
from transforms3d.euler import euler2mat, quat2euler
from transforms3d.quaternions import quat2mat

from robel.simulation.sim_scene import SimScene


class TrackerGroupConfig:
    """Group configuration for a TrackerComponent."""

    def __init__(self,
                 sim_scene: SimScene,
                 element_name: Optional[str] = None,
                 element_type: Optional[str] = None,
                 qpos_indices: Optional[Iterable[int]] = None,
                 qvel_indices: Optional[Iterable[int]] = None,
                 sim_observation_noise: Optional[float] = None):
        """Initializes a group configuration for a TrackerComponent.

        Args:
            sim_scene: The simulation, used for validation purposes.
            element_name: The name of the element to use for tracking in
                simulation.
            element_type: The type of the element as defined in the XML.
                Should be one of `site`, `body`, `geom`, or `joint`. If this is
                `joint`, `qpos_indices` and `qvel_indices` should be
                provided.
            qpos_indices: The indices into `MjData.qpos` to read for the
                joint element position and rotation.
            qvel_indices: The indices into `MjData.qvel` to read for the joint
                element velocity. This defaults to `qpos_indices`.
            sim_observation_noise: The range of the observation noise (in
                meters) to apply to the state in simulation.
        """
        self.element_type = element_type
        if self.element_type not in ['site', 'body', 'geom', 'joint']:
            raise ValueError('Unknown element type %s' % self.element_type)

        self.element_name = element_name
        self.element_id = None
        self.element_attr = None
        self.qpos_indices = None
        self.qvel_indices = None
        self._is_euler = False

        if self.element_type == 'joint':
            if qpos_indices is None:
                raise ValueError('Must provided qpos_indices for joints.')
            # Ensure that the qpos indices are valid.
            nq = sim_scene.model.nq
            assert all(-nq <= i < nq for i in qpos_indices), \
                'All qpos indices must be in [-{}, {}]'.format(nq, nq - 1)
            self.qpos_indices = np.array(qpos_indices, dtype=int)

            if len(self.qpos_indices) == 6:
                self._is_euler = True
            elif len(self.qpos_indices) != 7:
                raise ValueError('qpos_indices must be 6 or 7 elements.')

            if qvel_indices is None:
                if not self._is_euler:
                    raise ValueError(
                        'qvel_indices must be provided for free joints.')
                qvel_indices = qpos_indices

            # Ensure that the qvel indices are valid.
            nv = sim_scene.model.nv
            assert all(-nv <= i < nv for i in qvel_indices), \
                'All qvel indices must be in [-{}, {}]'.format(nv, nv - 1)
            self.qvel_indices = np.array(qvel_indices, dtype=int)
        else:
            self.element_attr = (lambda obj, attr_name: getattr(
                obj, self.element_type + '_' + attr_name))
            self.element_id = self.element_attr(sim_scene.model, 'name2id')(
                element_name)

        self.sim_observation_noise = sim_observation_noise

    def get_pos(self, sim_scene: SimScene) -> np.ndarray:
        """Returns the cartesian position of the element."""
        if self.qpos_indices is not None:
            return sim_scene.data.qpos[self.qpos_indices[:3]]
        return self.element_attr(sim_scene.data, 'xpos')[self.element_id, :]

    def get_rot(self, sim_scene: SimScene) -> np.ndarray:
        """Returns the (3x3) rotation matrix of the element."""
        if self.qpos_indices is not None:
            qpos = sim_scene.data.qpos[self.qpos_indices[3:]]
            if self._is_euler:
                return euler2mat(*qpos, axes='rxyz')
            return quat2mat(qpos)
        return self.element_attr(sim_scene.data,
                                 'xmat')[self.element_id].reshape((3, 3))

    def get_vel(self, sim_scene: SimScene) -> np.ndarray:
        """Returns the cartesian velocity of the element."""
        if self.qvel_indices is not None:
            return sim_scene.data.qvel[self.qvel_indices[:3]]
        raise NotImplementedError('Cartesian velocity is not supported for ' +
                                  self.element_type)

    def get_angular_vel(self, sim_scene: SimScene) -> np.ndarray:
        """Returns the angular velocity (x, y, z) of the element."""
        if self.qvel_indices is not None:
            return sim_scene.data.qvel[self.qvel_indices[3:]]
        raise NotImplementedError('Angular velocity is not supported for ' +
                                  self.element_type)

    def set_pos(self, sim_scene: SimScene, pos: np.ndarray):
        """Sets the cartesian position of the element."""
        if self.qpos_indices is not None:
            sim_scene.data.qpos[self.qpos_indices[:len(pos)]] = pos
            return
        self.element_attr(sim_scene.model,
                          'pos')[self.element_id, :len(pos)] = pos

    def set_rot_quat(self, sim_scene: SimScene, quat: np.ndarray):
        """Sets the cartesian position of the element."""
        if self.qpos_indices is not None:
            qpos = quat
            if self._is_euler:
                qpos = quat2euler(quat, axes='rxyz')
            sim_scene.data.qpos[self.qpos_indices[3:]] = qpos
            return
        self.element_attr(sim_scene.model, 'quat')[self.element_id, :] = quat
