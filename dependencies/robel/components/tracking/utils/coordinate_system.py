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

"""Coordinate system for pose-related logic."""

import collections
from typing import Optional, Union

import numpy as np
from transforms3d.quaternions import quat2mat

from robel.components.tracking.tracker import TrackerState

ObjectId = Union[str, int]


class CoordinateSystem:
    """Stores the most recently returned values for device poses."""

    def __init__(self):
        # Coordinate system transformation to apply to all poses.
        self._coordinate_transform = None

        # Global transformation to apply to all poses.
        self._global_translation = None
        self._global_rotation = None

        # Per-object transforms.
        self._local_translations = {}
        self._local_rotations = {}

        # Per-object state cache.
        self._state_cache = collections.defaultdict(TrackerState)

    def set_coordinate_transform(self, transform: np.ndarray):
        """Sets the coordinate transform."""
        if transform.shape != (3, 3):
            raise ValueError('')

    def set_global_transform(self, translation: np.ndarray,
                             rotation: np.ndarray):
        """Sets the global transform."""
        trans, rot = self._check_transform(translation, rotation)
        self._global_translation = trans
        self._global_rotation = rot

    def set_local_transform(self,
                            object_id: ObjectId,
                            translation: Optional[np.ndarray] = None,
                            rotation: Optional[np.ndarray] = None):
        """Sets the local transform for the given object."""
        trans, rot = self._check_transform(translation, rotation)
        if trans is not None:
            self._local_translations[object_id] = trans
        if rot is not None:
            self._local_rotations[object_id] = rot

    def record_state(self,
                     object_id: ObjectId,
                     state: TrackerState,
                     ignore_object_transform: bool = False) -> TrackerState:
        """Records and transforms the given state."""
        state = self.transform_raw_state(state)
        # Add to the state cache.
        cached_state = self._state_cache[object_id]
        for prop in ('pos', 'rot', 'vel', 'angular_vel'):
            new_value = getattr(state, prop)
            if new_value is not None:
                setattr(cached_state, prop, new_value)
            else:
                # Get the cached value.
                setattr(state, prop, getattr(cached_state, prop))
        if not ignore_object_transform:
            state = self.transform_object_state(object_id, state)
        return state

    def get_cached_state(self, object_id: ObjectId) -> TrackerState:
        """Returns the cached state for the object."""
        return self._state_cache[object_id]

    def transform_raw_state(self, state: TrackerState) -> TrackerState:
        """Transforms the given state to the coordinate system."""
        pos = state.pos
        rot = state.rot
        vel = state.vel
        angular_vel = state.angular_vel
        if self._coordinate_transform:
            if pos is not None:
                pos = np.matmul(self._coordinate_transform, pos)
            if rot is not None:
                rot = np.matmul(self._coordinate_transform, rot)
                rot = np.matmul(rot, np.transpose(self._coordinate_transform))
            if vel is not None:
                vel = np.matmul(self._coordinate_transform, vel)
            if angular_vel is not None:
                angular_vel = np.matmul(self._coordinate_transform, angular_vel)
        return TrackerState(pos=pos, rot=rot, vel=vel, angular_vel=angular_vel)

    def transform_object_state(self, object_id: ObjectId, state: TrackerState):
        """Transforms the given object state to the coordinate system."""
        pos = state.pos
        rot = state.rot
        vel = state.vel
        angular_vel = state.angular_vel
        if pos is not None:
            if self._global_translation is not None:
                pos = pos + self._global_translation
            if object_id in self._local_translations:
                pos = pos + self._local_translations[object_id]

        rotations = [
            self._global_rotation,
            self._local_rotations.get(object_id)
        ]
        for rotation in rotations:
            if rotation is None:
                continue
            if pos is not None:
                pos = np.matmul(rotation, pos)
            if rot is not None:
                rot = np.matmul(rotation, rot)
            if vel is not None:
                vel = np.matmul(rotation, vel)
            if angular_vel is not None:
                angular_vel = np.matmul(rotation, angular_vel)
        return TrackerState(pos=pos, rot=rot, vel=vel, angular_vel=angular_vel)

    def _check_transform(self, translation: Optional[np.ndarray],
                         rotation: Optional[np.ndarray]):
        """Checks that the given translation and rotation are valid."""
        if translation is None:
            pass
        elif translation.shape != (3,):
            raise ValueError('Expected translation to have shape 3')
        else:
            translation = translation.copy()
        if rotation is None:
            pass
        elif rotation.shape == (4,):
            rotation = quat2mat(rotation)
        elif rotation.shape == (3, 3):
            rotation = rotation.copy()
        else:
            raise ValueError('Expected rotation to have shape (3, 3)')
        return translation, rotation
