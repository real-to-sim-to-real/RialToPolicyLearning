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

"""Helper methods for Gym environment registration."""

import logging

from gym.envs import registration as gym_reg


def register(env_id: str, class_path: str, **kwargs):
    """Registers the given class path as a Gym environment.

    Args:
        env_id: The ID to register the environment as.
        class_path: The fully-qualified class path of the environment.
        **kwargs: Key-word arguments to pass to gym's register function.
    """
    if env_id in gym_reg.registry.env_specs:
        # This may happen during test discovery.
        logging.warning('Re-registering environment %s', env_id)
        del gym_reg.registry.env_specs[env_id]

    gym_reg.register(env_id, entry_point=class_path, **kwargs)
