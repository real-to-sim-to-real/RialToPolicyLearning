from polymetis import RobotInterface
from polymetis import GripperInterface
import torch

ip_addr = "173.16.0.1"
robot = RobotInterface(ip_address=ip_addr)
gripper = GripperInterface(ip_address=ip_addr)


robot.start_joint_impedance(Kq=torch.zeros(7), Kqd=torch.zeros(7), adaptive=False)
gripper.goto(0.08, 0.05, 0.01)

