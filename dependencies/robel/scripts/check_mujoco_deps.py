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

"""Checks if the given MuJoCo XML file has valid dependencies.

Example usage:
python -m robel.scripts.check_mujoco_deps path/to/mujoco.xml
"""

import argparse
import logging
import os

from robel.utils.resources import AssetBundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs=1, help='The MuJoCo XML to parse.')
    args = parser.parse_args()
    model_path = args.path[0]

    if not os.path.exists(model_path):
        raise ValueError('Path does not exist: ' + model_path)

    logging.basicConfig(level=logging.INFO)
    with AssetBundle(dry_run=True, verbose=True) as bundle:
        bundle.add_mujoco(model_path)


if __name__ == '__main__':
    main()
