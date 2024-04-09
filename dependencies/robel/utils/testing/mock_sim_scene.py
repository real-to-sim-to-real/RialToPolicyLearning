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

"""Mock SimScene for testing."""

import contextlib
from typing import Iterable, Optional, Tuple

from absl.testing.absltest import mock
import numpy as np


class MockMjData:
    """Mock container for Mujoco data."""

    def __init__(self, nq: int, nv: int, nu: int):
        self.qpos = np.zeros(nq, dtype=np.float32)
        self.qvel = np.zeros(nv, dtype=np.float32)
        self.qacc = np.zeros(nv, dtype=np.float32)
        self.ctrl = np.zeros(nu, dtype=np.float32)


class MockMjModel:
    """Mock container for a Mujoco model."""

    # Properties of the model, mapped to the size of the property.
    PROPERTIES = {
        'body': {
            'pos': 3,
            'quat': 4,
        },
        'geom': {
            'pos': 3,
            'quat': 4,
            'size': 3,
        },
        'site': {
            'pos': 3,
            'quat': 4,
            'size': 3,
        },
        'cam': {
            'pos': 3,
            'quat': 4,
        }
    }

    def __init__(self,
                 nq: int,
                 nv: Optional[int] = None,
                 nu: Optional[int] = None,
                 ctrl_range: Optional[Tuple[float]] = None,
                 body_names: Optional[Iterable[str]] = None,
                 geom_names: Optional[Iterable[str]] = None,
                 site_names: Optional[Iterable[str]] = None,
                 cam_names: Optional[Iterable[str]] = None):
        if nv is None:
            nv = nq
        if nu is None:
            nu = nq
        self.nq = nq
        self.nv = nv
        self.nu = nu
        self.body_names = body_names or []
        self.geom_names = geom_names or []
        self.site_names = site_names or []
        self.cam_names = cam_names or []

        self.data = MockMjData(nq, nv, nu)

        if ctrl_range is None:
            ctrl_range = (-1, 1)
        self.actuator_ctrlrange = np.tile(ctrl_range, (self.nu, 1))

        # Generate the properties.
        for parent_key, sub_properties in self.PROPERTIES.items():
            names = getattr(self, parent_key + '_names')
            element_count = len(names)
            # Add the count property.
            setattr(self, 'n' + parent_key, element_count)

            # Add mujoco_py's `*_name2id`  method.
            mapping = {name: i for i, name in enumerate(names)}

            def name2id(name: str, parent=parent_key, key_map=mapping):
                if name not in key_map:
                    raise ValueError('No {} exists with name {}'.format(
                        parent, name))
                return key_map[name]

            if parent_key == 'cam':  # mujoco-py is inconsistent for camera.
                fn_name = 'camera_name2id'
            else:
                fn_name = parent_key + '_name2id'
            setattr(self, fn_name, name2id)

            # Create the child-property arrays.
            for child_key, size in sub_properties.items():
                attr_name = '{}_{}'.format(parent_key, child_key)
                setattr(self, attr_name,
                        np.zeros((element_count, size), dtype=np.float32))


class MockSimScene:
    """Mock object that implements the SimScene interface."""

    @staticmethod
    def create(*unused_args, **unused_kwargs):
        raise NotImplementedError('`patch_sim_scene` must be called.')

    def __init__(self, *args, step_duration: float = 1, **kwargs):
        """Initializes a new mock SimScene."""
        self.sim = mock.Mock()
        self.sim.model = MockMjModel(*args, **kwargs)

        self.model = self.sim.model
        self.sim.data = self.model
        self.data = self.model.data

        self.step_duration = step_duration

        self.close = mock.Mock()
        self.advance = mock.Mock()

        self.renderer = mock.Mock()
        self.renderer.render_offscreen = lambda w, h, **_: np.zeros((w, h))


@contextlib.contextmanager
def patch_sim_scene(module_path: str, **kwargs):
    """Patches the SimScene class in the given module.

    Args:
        module_path: The path to the SimScene class to mock.
        **kwargs: Arguments passed to MockSimScene when `SimScene.create` is
            called.
    """
    with mock.patch(module_path, MockSimScene) as mock_sim_cls:
        mock_sim_cls.create = lambda *_args, **_kwargs: MockSimScene(**kwargs)
        yield mock_sim_cls
