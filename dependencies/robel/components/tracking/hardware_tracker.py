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

"""Implementation of TrackerComponent using OpenVR to track devices."""

import abc
import logging
from typing import Dict, Optional, Sequence, Union

import numpy as np
from transforms3d.euler import euler2quat, euler2mat, quat2euler
from transforms3d.quaternions import quat2mat, qconjugate

from robel.components.tracking.tracker import (
    TrackerComponent, TrackerGroupConfig, TrackerState)
from robel.components.tracking.utils.coordinate_system import (
    CoordinateSystem)
from robel.utils.math_utils import average_quaternions

DeviceId = Union[int, str]


class HardwareTrackerGroupConfig(TrackerGroupConfig):
    """Stores group configuration for a VrTrackerComponent."""
    def __init__(self,
                 *args,
                 device_identifier: Optional[DeviceId] = None,
                 is_origin: bool = False,
                 tracked_position_offset: Optional[Sequence[float]] = None,
                 tracked_rotation_offset: Optional[Sequence[float]] = None,
                 mimic_in_sim: bool = False,
                 mimic_ignore_z_axis: bool = False,
                 mimic_ignore_rotation: bool = False,
                 **kwargs):
        """Initializes a new configuration for a HardwareTrackerComponent group.

        Args:
            device_identifier: The device index or device serial string of the
                tracking device.
            is_origin: If True, the (0, 0, 0) world position is inferred to be
                at this group's location.
            tracked_position_offset: The offset to add to the tracked positions.
            tracked_rotation_offset: The offset to add to the tracked rotations.
            mimic_in_sim: If True, updates the simulation sites with the tracked
                positions.
            mimic_ignore_z_axis: If True, the simulation site is only updated
                with the x and y position.
            mimic_ignore_rotation: If True, the simulation site is not updated
                with the rotation.
        """
        super().__init__(*args, **kwargs)
        self.device_identifier = device_identifier
        self.is_origin = is_origin
        self.mimic_in_sim = mimic_in_sim
        self.mimic_ignore_z_axis = mimic_ignore_z_axis
        self.mimic_ignore_rotation = mimic_ignore_rotation

        self.tracked_position_offset = None
        if tracked_position_offset is not None:
            tracked_position_offset = np.array(
                tracked_position_offset, dtype=np.float32)
            assert tracked_position_offset.shape == (
                3,), tracked_position_offset.shape
            self.tracked_position_offset = tracked_position_offset

        self.tracked_rotation_offset = None
        if tracked_rotation_offset is not None:
            tracked_rotation_offset = np.array(
                tracked_rotation_offset, dtype=np.float32)
            assert tracked_rotation_offset.shape == (
                3,), tracked_rotation_offset.shape
            self.tracked_rotation_offset = euler2quat(
                *tracked_rotation_offset, axes='rxyz')


class HardwareTrackerComponent(TrackerComponent):
    """Component for reading tracking data from a HTC Vive."""
    def __init__(self, *args, **kwargs):
        """Create a new hardware tracker component."""
        super().__init__(*args, **kwargs)
        self._coord_system = CoordinateSystem()
        self._plot = None

    @property
    def is_hardware(self) -> bool:
        """Returns True if this is a hardware component."""
        return True

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return HardwareTrackerGroupConfig(self.sim_scene, **config_kwargs)

    def _get_group_states(
            self,
            configs: Sequence[HardwareTrackerGroupConfig],
            raw_states: bool = False,
    ) -> Sequence[TrackerState]:
        """Returns the TrackerState for the given groups.

        Args:
            configs: The group configurations to retrieve the states for.

        Returns:
            A list of TrackerState(timestamp, pos, quat, euler).
        """
        self._refresh_poses()
        simulation_changed = False
        states = []
        for config in configs:
            if config.device_identifier is None:
                # Fall back to simulation site.
                states.append(self._get_element_state(config))
                continue
            state = self._get_tracker_state(
                config.device_identifier, ignore_device_transform=raw_states)

            # Mimic the site in simulation if needed.
            if config.mimic_in_sim:
                mimic_state = TrackerState(
                    pos=state.pos,
                    rot=None if config.mimic_ignore_rotation else state.rot)
                simulation_changed |= self._set_element_state(
                    mimic_state,
                    config,
                    ignore_z_axis=config.mimic_ignore_z_axis)

            states.append(state)

        if simulation_changed:
            self.sim_scene.sim.forward()
        return states

    def set_state(self, state_groups: Dict[str, TrackerState]):
        """Sets the tracker to the given initial state.

        Args:
            state_groups: A mapping of control group name to desired position
                and velocity.
        """
        origin_device_id = None
        device_positions = {}
        device_rotations = {}
        ignored_group_positions = []

        for group_name, state in state_groups.items():
            config = self.get_config(group_name)
            device_id = config.device_identifier
            if device_id is None:
                continue

            pos_offset = config.tracked_position_offset
            quat_offset = config.tracked_rotation_offset

            # Only respect the set position for the origin device.
            if state.pos is not None:
                assert state.pos.shape == (3,), state.pos
                if config.is_origin:
                    if origin_device_id is not None:
                        raise ValueError('Cannot have more than one origin.')
                    origin_device_id = device_id
                    if pos_offset is None:
                        pos_offset = state.pos
                    else:
                        pos_offset = pos_offset + state.pos
                else:
                    ignored_group_positions.append(
                        (group_name, device_id, state.pos))

            if state.rot is not None:
                logging.warning('Ignoring setting rotation for group: "%s"',
                                group_name)
            device_positions[device_id] = pos_offset
            device_rotations[device_id] = quat_offset

        # Set the coordinate system.
        self._update_coordinate_system(origin_device_id, device_positions,
                                       device_rotations)

        # Log any ignored positions.
        if ignored_group_positions:
            self._refresh_poses()
            for group_name, device_id, desired_pos in ignored_group_positions:
                state = self._get_tracker_state(device_id)
                logging.warning(
                    'Ignored setting position for group: "%s" - '
                    'Desired position: %s, Actual position: %s', group_name,
                    str(desired_pos.tolist()), str(state.pos.tolist()))

    def _update_coordinate_system(self,
                                  origin_device_id: Optional[DeviceId],
                                  device_positions: Dict[DeviceId, np.ndarray],
                                  device_rotations: Dict[DeviceId, np.ndarray],
                                  num_samples: int = 10):
        """Updates the coordinate system origin."""
        if origin_device_id:
            # Collect samples for the devices.
            pos_samples = []
            quat_samples = []
            for _ in range(num_samples):
                self._refresh_poses()
                state = self._get_tracker_state(
                    origin_device_id, ignore_device_transform=True)
                pos_samples.append(state.pos)
                quat_samples.append(state.rot_quat)
            global_translation = -np.mean(pos_samples, axis=0)
            global_rotation = qconjugate(average_quaternions(quat_samples))
            origin_rx, origin_ry, origin_rz = quat2euler(
                global_rotation, axes='rxyz')
            logging.info('Have origin rotation: %1.2f %1.2f %1.2f', origin_rx,
                         origin_ry, origin_rz)
            self._coord_system.set_global_transform(global_translation,
                                                    euler2mat(0, 0, origin_rz))

        for device_id, position in device_positions.items():
            self._coord_system.set_local_transform(
                device_id, translation=position)
        for device_id, rotation in device_rotations.items():
            self._coord_system.set_local_transform(device_id, rotation=rotation)

    def _get_tracker_state(
            self,
            device_id: DeviceId,
            ignore_device_transform: bool = False,
    ) -> TrackerState:
        """Returns the tracker state in the coordinate system."""
        state = self._get_raw_tracker_state(device_id)
        state = self._coord_system.record_state(
            device_id, state, ignore_object_transform=ignore_device_transform)
        return state

    def show_plot(self, auto_update: bool = False, frame_rate: int = 10):
        """Displays a plot that shows the current tracked positions.

        Args:
            auto_update: If True, queries and updates the plot with new pose
                data at the given frame rate.
            frame_rate: The frequency at which the plot is refreshed.
        """
        if self._plot and self._plot.is_open:
            return
        from robel.utils.plotting import AnimatedPlot
        self._plot = AnimatedPlot(bounds=[-5, 5, -5, 5])

        device_ids = [(name, config.device_identifier)
                      for name, config in self.groups.items()]

        data = np.zeros((len(device_ids), 2), dtype=np.float32)
        scatter = self._plot.ax.scatter(
            data[:, 0], data[:, 1], cmap='jet', edgecolor='k')
        self._plot.add(scatter)

        # Make annotations
        labels = []
        for i, (name, device_id) in enumerate(device_ids):
            text = '{} ({})'.format(name, device_id)
            label = self._plot.ax.annotate(text, xy=(data[i, 0], data[i, 1]))
            labels.append(label)
            self._plot.add(label)

        def update():
            if auto_update:
                self._refresh_poses()

            for i, (name, device_id) in enumerate(device_ids):
                info = ['{} ({})'.format(name, device_id)]
                state = self._get_tracker_state(device_id)
                pos = state.pos
                euler = state.rot_euler
                if pos is not None:
                    data[i, 0] = pos[0]
                    data[i, 1] = pos[1]
                    labels[i].set_position((pos[0], pos[1] + 0.2))
                    info.append('Tx:{:.2f} Ty:{:.2f} Tz:{:.2f}'.format(*pos))
                if euler is not None:
                    info.append('Rx:{:.2f} Ry:{:.2f} Rz:{:.2f}'.format(*euler))
                labels[i].set_text('\n'.join(info))
            scatter.set_offsets(data)

        self._plot.update_fn = update
        self._plot.show(frame_rate=frame_rate)

    @abc.abstractmethod
    def _refresh_poses(self):
        """Refreshes the pose state."""

    @abc.abstractmethod
    def _get_raw_tracker_state(self, device_id: DeviceId) -> TrackerState:
        """Returns the tracker state."""
