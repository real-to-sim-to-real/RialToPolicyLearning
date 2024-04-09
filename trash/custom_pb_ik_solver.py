import time

import numpy as np

from airobot import Robot
from airobot import log_warn
from airobot.utils.common import euler2quat


class CustomPBIKSolver:
    def __init__(self):
        np.set_printoptions(precision=4, suppress=True)
        self.robot = Robot('franka')
        success = self.robot.arm.go_home()
        self.num_actions = 8

    def solve_ik(self, base_position, base_orientation, target_position, target_orientation):
        ik_kwargs = dict(ns=True)
        eef_step = 0.025
        print("target target_position", target_position)
        print(self.robot.arm.get_ee_pose())
        # target_position = [0.5526, -0.0190,  0.3149+1]
        target_position = target_position + np.array([0,0,1])
        self.robot.arm.set_ee_pose(pos=target_position, eef_step=eef_step, **dict(ik_kwargs=ik_kwargs))
        # self.robot.arm.move_ee_xyz(target_position, eef_step=eef_step, **dict(ik_kwargs=ik_kwargs))
        return self.robot.arm.get_jpos()

    def solve_all_robots(self,robot_frame, actions):
        all_joints = []
        
        for i in range(robot_frame.shape[0]):
            # import IPython
            # IPython.embed()
            orientation = robot_frame[i,3:7].copy()
            # p1  = orientation[0]
            # orientation[:3] = orientation[1:]
            # orientation[-1] = p1
            print("orientation", orientation)
            print("actions", actions)
            # mid_or = orientation[0]
            # orientation[0] = orientation[-1]
            # orientation[-1] = mid_or
            ori_act = actions[i, 3:7].copy()
            # pi = 3.1415
            # ori_act = p.getQuaternionFromEuler([0,0,pi/2])
            # actions[i,:3] = [0.1,0.1,0.1]
            p1  = ori_act[0]
            ori_act[:3] = ori_act[1:]
            ori_act[-1] = p1
            # mid_or = ori_act[-1]
            # ori_act[-1] = ori_act[0]
            # ori_act[0] = mid_or
            # ori_act = [0,0,0,1]
            # orientation = [0,0,0,1]
            joints = self.solve_ik(base_position=robot_frame[i,:3], base_orientation=orientation, target_position=actions[i,:3], target_orientation=ori_act)
            all_joints.append(joints)
            # self.move_to_joints(joints)

        return np.array(all_joints)
    


    def random_movements(self):
        robot_frame = np.array([[0,0,0,0,0,0,1]])
        action = np.array([[0.4907, 0.0238, 1.1702, 0.9931, -0.0286,  0.0909, -0.0691]])
        # pi =3.1415
        # action[0,3:] = np.array(p.getQuaternionFromEuler([0,0,pi]))
        for i in range(100):
            action[0,0] += 0.05
            self.solve_all_robots(robot_frame, action)
            time.sleep(1)