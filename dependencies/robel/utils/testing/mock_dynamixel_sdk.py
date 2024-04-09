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

"""Mocks the DynamixelSDK Python API."""
# DynamixelSDK conforms to a different naming convention.
# pylint: disable=invalid-name

import sys
from typing import Iterable

from absl.testing.absltest import mock
import numpy as np

from robel.components.robot.dynamixel_client import ADDR_TORQUE_ENABLE

CONTROL_TABLE_SIZE = 148


class MockDynamixelSdk:
    """Mock class for the DynamixelSDK."""

    def __init__(self):
        self.available_ports = set()
        self.used_ports = set()
        self.devices = {}
        self.has_opened_port = False

        # dynamixel_sdk constants.
        self.COMM_SUCCESS = True

    def create_device(self, port: str, motor_ids: Iterable[int]):
        """Creates a fake device."""
        assert port not in self.available_ports
        self.available_ports.add(port)
        self.devices[port] = {
            motor_id: np.zeros(CONTROL_TABLE_SIZE, dtype=np.uint8)
            for motor_id in motor_ids
        }

    def get_enabled_motors(self, port: str) -> Iterable[int]:
        """Returns the enabled motor IDs for the given port."""
        motor_ids = []
        for motor_id, control_table in self.devices[port].items():
            if control_table[ADDR_TORQUE_ENABLE] == 1:
                motor_ids.append(motor_id)
        return motor_ids

    def PortHandler(self, port: str):
        """Returns a mock port handler."""
        if port not in self.available_ports:
            raise ValueError('Unknown port: {}'.format(port))
        if port in self.used_ports:
            raise ValueError('Port in use: {}'.format(port))
        # dynamixel_sdk has an undocumented behavior that all PortHandlers must
        # be created before openPort is called, or else port numbers are reused.
        # Enforce this in tests by erroring.
        if self.has_opened_port:
            raise ValueError(
                'Must create all PortHandlers before openPort is called')

        self.used_ports.add(port)

        handler = mock.Mock(spec=[])

        def openPort():
            self.has_opened_port = True
            handler.is_open = True
            return not handler.faulty

        def closePort():
            handler.is_open = False

        handler.is_open = False
        handler.is_using = False
        handler.faulty = False
        handler.openPort = mock.Mock(side_effect=openPort)
        handler.closePort = mock.Mock(side_effect=closePort)
        handler.setBaudRate = mock.Mock(
            side_effect=lambda _: not handler.faulty)
        handler.device = self.devices[port]
        return handler

    def PacketHandler(self, protocol_version: int = 2.0):
        """Returns a mock port handler."""
        handler = mock.Mock(spec=[])
        handler.protocol_version = protocol_version
        handler.faulty_comm = False
        handler.faulty_packets = False

        def write1ByteTxRx(port_handler, motor_id: int, address: int,
                           value: int):
            if motor_id not in port_handler.device:
                raise ValueError('Invalid motor ID: {}'.format(motor_id))
            port_handler.device[motor_id][address] = value
            return not handler.faulty_comm, not handler.faulty_packets

        handler.write1ByteTxRx = mock.Mock(side_effect=write1ByteTxRx)
        handler.getTxRxResult = mock.Mock(
            side_effect=lambda success: 'Error!' if not success else None)
        handler.getRxPacketError = handler.getTxRxResult
        return handler

    def GroupBulkRead(self, port_handler, unused_packet_handler):
        """Returns a mock bulk read operation."""
        op = mock.Mock(spec=[])
        op.params = {}

        device = port_handler.device

        def addParam(motor_id: int, address: int, size: int):
            if motor_id not in device or motor_id in op.params:
                return False
            assert address + size <= device[motor_id].size
            op.params[motor_id] = (address, size)
            return True

        op.faulty = False
        op.unavailable_ids = set()
        op.addParam = mock.Mock(side_effect=addParam)
        op.txRxPacket = mock.Mock(side_effect=lambda: not op.faulty)

        def isAvailable(motor_id: int, address: int, size: int):
            assert motor_id in op.params
            if motor_id in op.unavailable_ids:
                return False
            assert address + size <= device[motor_id].size
            return True

        op.isAvailable = mock.Mock(side_effect=isAvailable)

        def getData(motor_id: int, address: int, size: int):
            assert motor_id in op.params
            assert motor_id in device
            assert size in (1, 2, 4)
            data = device[motor_id][address:address + size]
            return int.from_bytes(data.tobytes(), byteorder='little')

        op.getData = mock.Mock(side_effect=getData)
        return op

    def GroupSyncWrite(self, port_handler, unused_packet_handler, address: int,
                       size: int):
        """Returns a mock sync write operation."""
        op = mock.Mock(spec=[])
        op.params = set()

        device = port_handler.device

        def addParam(motor_id: int, value: bytes):
            if motor_id not in device or motor_id in op.params:
                return False
            if len(value) != size:
                raise ValueError('Incorrect size for value: {}'.format(value))
            device[motor_id][address:address + size] = list(value)
            op.params.add(motor_id)
            return True

        op.txPacket = mock.Mock(return_value=True)
        op.addParam = mock.Mock(side_effect=addParam)
        op.clearParam = mock.Mock(side_effect=op.params.clear)
        return op


def patch_dynamixel(**devices):
    """Decorator that patches the DynamixelSDK for the function context."""

    def decorator(fn):
        def wrapped_fn(*args):
            sdk = MockDynamixelSdk()
            for key, motor_ids in devices.items():
                sdk.create_device(key, motor_ids)
            sys.modules['dynamixel_sdk'] = sdk
            fn(*args, sdk)
            del sys.modules['dynamixel_sdk']

        return wrapped_fn

    return decorator
