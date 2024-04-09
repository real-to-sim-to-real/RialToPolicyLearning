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

"""Runs a softlearning policy.

Example usage:
python -m robel.scripts.enjoy_softlearning --device /dev/ttyUSB0

This runs the DClawTurnFixed-v0 environment by default. To run other
environments, pass in the environment name with `-e/--env_name`

python -m robel.scripts.enjoy_softlearning \
    --env_name DClawScrewFixed-v0 \
    --device /dev/ttyUSB0
"""

import argparse
import os
import pickle

import numpy as np

from robel.scripts.rollout import rollout_script

POLICY_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 'data/softlearning')

DEFAULT_POLICY_FORMAT = os.path.join(POLICY_DIR, '{}.pkl')


def policy_factory(args: argparse.Namespace):
    """Creates the policy."""
    # Get default policy path from the environment name.
    policy_path = args.policy
    if not policy_path:
        policy_path = DEFAULT_POLICY_FORMAT.format(args.env_name)

    # Load the policy
    with open(policy_path, 'rb') as f:
        policy_data = pickle.load(f)
        policy = policy_data['policy']
        policy.set_weights(policy_data['weights'])

    def policy_fn(obs):
        with policy.set_deterministic(True):
            actions = policy.actions_np(np.expand_dims(obs, axis=0))
        return actions[0]

    return policy_fn


if __name__ == '__main__':
    rollout_script(policy_factory=policy_factory, add_policy_arg=True)
