from polymetis import RobotInterface
import torch
import time

ip_addr = "173.16.0.1"
robot = RobotInterface(ip_address=ip_addr)

robot.start_joint_impedance(Kx=torch.zeros(7), Kxd=torch.zeros(7), adaptive=False)

while True:
    joint_pos = robot.get_joint_positions()
    print("joint pos", joint_pos)
    time.sleep(5)