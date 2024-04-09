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

"""Unit tests for DynamixelClient."""

from absl.testing import absltest

from robel.components.robot.dynamixel_client import DynamixelClient
from robel.utils.testing.mock_dynamixel_sdk import patch_dynamixel


class DynamixelClientTest(absltest.TestCase):
    """Unit test class for DynamixelClient."""

    @patch_dynamixel(test=[1, 2, 3, 4])
    def test_connect(self, sdk):
        client = DynamixelClient([1, 2, 3, 4], port='test')
        self.assertFalse(client.is_connected)

        client.connect()
        self.assertIn('test', sdk.used_ports)
        self.assertListEqual(sdk.get_enabled_motors('test'), [1, 2, 3, 4])
        client.disconnect()

    @patch_dynamixel(test=[1, 2, 3, 4])
    def test_torque_enabled(self, sdk):
        client = DynamixelClient([1, 2, 3, 4], port='test')
        client.connect()
        self.assertListEqual(sdk.get_enabled_motors('test'), [1, 2, 3, 4])

        client.set_torque_enabled([1, 3], False)
        self.assertListEqual(sdk.get_enabled_motors('test'), [2, 4])

        client.set_torque_enabled([1, 2], True)
        self.assertListEqual(sdk.get_enabled_motors('test'), [1, 2, 4])

        client.disconnect()
        self.assertListEqual(sdk.get_enabled_motors('test'), [])


if __name__ == '__main__':
    absltest.main()
