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

"""Unit tests for MockDynamixelSdk."""

from absl.testing import absltest

from robel.utils.testing.mock_dynamixel_sdk import patch_dynamixel


class MockDynamixelSdkTest(absltest.TestCase):
    """Tests MockDynamixelSdk."""

    @patch_dynamixel(test=[1, 2, 3, 4])
    def test_port_handler(self, sdk):
        handler = sdk.PortHandler('test')

        self.assertTrue(handler.openPort())
        self.assertTrue(handler.setBaudRate(1000000))
        handler.closePort()

    @patch_dynamixel(test=[1, 2, 3, 4], test1=[1, 2])
    def test_port_handler_fault(self, sdk):
        with self.assertRaises(ValueError):
            sdk.PortHandler('another')

        handler = sdk.PortHandler('test')

        with self.assertRaises(ValueError):
            sdk.PortHandler('test')

        handler.faulty = True
        self.assertFalse(handler.openPort())

        with self.assertRaises(ValueError):
            sdk.PortHandler('test1')

        self.assertFalse(handler.setBaudRate(1000000))
        self.assertIn('test', sdk.used_ports)
        self.assertNotIn('test1', sdk.used_ports)

    @patch_dynamixel(test1=[1, 2, 3, 4], test2=[1, 2, 3, 4, 5, 6])
    def test_port_handler_multi(self, sdk):
        handler1 = sdk.PortHandler('test1')
        handler2 = sdk.PortHandler('test2')

        self.assertTrue(handler1.openPort())
        self.assertTrue(handler2.openPort())

    @patch_dynamixel(test=[1])
    def test_packet_handler(self, sdk):
        port = sdk.PortHandler('test')
        packet = sdk.PacketHandler()

        packet.write1ByteTxRx(port, 1, 64, 2)
        self.assertEqual(sdk.devices['test'][1][64], 2)

    @patch_dynamixel(test=[1, 2, 3, 4])
    def test_read_write(self, sdk):
        motor_ids = [1, 2, 3, 4]
        port = sdk.PortHandler('test')
        packet = sdk.PacketHandler()

        self.assertTrue(port.openPort())
        self.assertTrue(port.setBaudRate(1000000))

        reader = sdk.GroupBulkRead(port, packet)
        writer = sdk.GroupSyncWrite(port, packet, 32, 4)

        for mid in motor_ids:
            self.assertTrue(reader.addParam(mid, 32, 4))

        for mid in motor_ids:
            self.assertTrue(reader.isAvailable(mid, 32, 4))

        for mid in motor_ids:
            self.assertEqual(reader.getData(mid, 32, 4), 0)

        payload = 12345678
        for mid in motor_ids:
            self.assertTrue(writer.addParam(mid, payload.to_bytes(4, 'little')))

        self.assertTrue(writer.txPacket())

        for mid in motor_ids:
            self.assertTrue(reader.getData(mid, 32, 4), payload)


if __name__ == '__main__':
    absltest.main()
