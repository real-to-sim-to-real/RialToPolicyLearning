import six
import unittest

import torch as th
import rlutil.torch as torch
import rlutil.torch.nn as nn

class DeviceTest(unittest.TestCase):
    def testCPUDevice(self):
        torch.set_gpu(False)
        device_name = str(torch.ones(1).device)
        self.assertTrue(device_name.startswith('cpu'))

    def testGPUDevice(self):
        torch.set_gpu(True)
        device_name = str(torch.ones(1).device)
        self.assertTrue(device_name.startswith('cuda' if th.cuda.is_available() else 'cpu'))


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.linear = torch.nn.Linear(3, 3)


class ModuleTest(unittest.TestCase):
    def testCPUModule(self):
        torch.set_gpu(False)
        network = TestModule()
        device_name = str(network.linear.weight.device)
        self.assertTrue(device_name.startswith('cpu'))

    def testGPUDevice(self):
        torch.set_gpu(True)
        network = TestModule()
        device_name = str(network.linear.weight.device)
        self.assertTrue(device_name.startswith('cuda' if th.cuda.is_available() else 'cpu'))


if __name__ == "__main__":
    unittest.main()
