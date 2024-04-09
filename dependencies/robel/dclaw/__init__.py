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

"""Gym environment registration for DClaw environments."""

from robel.utils.registration import register

#===============================================================================
# Pose tasks
#===============================================================================

# Default number of steps per episode.
_POSE_EPISODE_LEN = 80  # 80*20*2.5ms = 4s

register(
    env_id='DClawPoseFixed-v0',
    class_path='robel.dclaw.pose:DClawPoseFixed',
    max_episode_steps=_POSE_EPISODE_LEN)

register(
    env_id='DClawPoseRandom-v0',
    class_path='robel.dclaw.pose:DClawPoseRandom',
    max_episode_steps=_POSE_EPISODE_LEN)

register(
    env_id='DClawPoseRandomDynamics-v0',
    class_path='robel.dclaw.pose:DClawPoseRandomDynamics',
    max_episode_steps=_POSE_EPISODE_LEN)

#===============================================================================
# Turn tasks
#===============================================================================

# Default number of steps per episode.
_TURN_EPISODE_LEN = 40  # 40*40*2.5ms = 4s

register(
    env_id='DClawTurnFixed-v0',
    class_path='robel.dclaw.turn:DClawTurnFixed',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandom-v0',
    class_path='robel.dclaw.turn:DClawTurnRandom',
    max_episode_steps=_TURN_EPISODE_LEN)

register(
    env_id='DClawTurnRandomDynamics-v0',
    class_path='robel.dclaw.turn:DClawTurnRandomDynamics',
    max_episode_steps=_TURN_EPISODE_LEN)


#===============================================================================
# Screw tasks
#===============================================================================

# Default number of steps per episode.
_SCREW_EPISODE_LEN = 80  # 80*40*2.5ms = 8s

register(
    env_id='DClawScrewFixed-v0',
    class_path='robel.dclaw.screw:DClawScrewFixed',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandom-v0',
    class_path='robel.dclaw.screw:DClawScrewRandom',
    max_episode_steps=_SCREW_EPISODE_LEN)

register(
    env_id='DClawScrewRandomDynamics-v0',
    class_path='robel.dclaw.screw:DClawScrewRandomDynamics',
    max_episode_steps=_SCREW_EPISODE_LEN)
