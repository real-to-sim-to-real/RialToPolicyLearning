from polymetis import RobotInterface
import torch

ip_addr = "173.16.0.1"
robot = RobotInterface(ip_address=ip_addr)

reset_joints = torch.tensor([-2.2819e-03, -1.3304e+00,  6.5887e-03, -2.6921e+00,  1.0691e-02,
         1.4201e+00,  7.2247e-01])

robot.move_to_joint_positions(reset_joints)