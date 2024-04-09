import pybullet as p
import pybullet_data
import numpy as np
import time
class CustomPBIKSolver:
    def __init__(self):
        clid = p.connect(p.SHARED_MEMORY)
        if (clid < 0):
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,0)
        p.setRealTimeSimulation(1)
        self.franka_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
        # import IPython
        # IPython.embed()
        # self.lower_limits
        # self.upper_limits
        # self.joint_ranges
        # self.rest_poses
        # self.joint_damping
        num_joints = p.getNumJoints(self.franka_id)
        print("Num joints", num_joints)
        self.end_effector_index = 8
        self.dof = p.getNumJoints(self.franka_id) -1
        self.joints = range(self.dof)
        self.num_actions = self.dof

    def solve_ik(self, base_position, base_orientation, target_position, target_orientation):
        p.resetBasePositionAndOrientation(self.franka_id, base_position, base_orientation)
        joint_poses = p.calculateInverseKinematics(self.franka_id,
                                                   self.end_effector_index,
                                                   target_position,
                                                   target_orientation,
                                                #    lowerLimits=self.lower_limits,
                                                #    upperLimits=self.upper_limits,
                                                #    jointRanges=self.joint_ranges,
                                                #    restPoses=self.rest_poses
                                                   )
        
        return list(joint_poses)
    
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
            # p1  = ori_act[0]
            # ori_act[:3] = ori_act[1:]
            # ori_act[-1] = p1
            # mid_or = ori_act[-1]
            # ori_act[-1] = ori_act[0]
            # ori_act[0] = mid_or
            # ori_act = [0,0,0,1]
            # orientation = [0,0,0,1]
            joints = self.solve_ik(base_position=robot_frame[i,:3], base_orientation=orientation, target_position=actions[i,:3], target_orientation=ori_act)
            all_joints.append(joints)
            self.move_to_joints(joints)

        return np.array(all_joints)
    
    def move_to_joints(self, joints):
        if len(joints) == 9:
            joints = joints + [0.,0.]
        
        p.setJointMotorControlArray(
            bodyIndex=self.franka_id,
            jointIndices=self.joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joints,
        )

        p.stepSimulation()

    def random_movements(self):
        robot_frame = np.array([[0,0,0,0,0,0,1]])
        action = np.array([[0,0,0,0,0,9.99999999e-01, 4.63267949e-05]])
        # pi =3.1415
        # action[0,3:] = np.array(p.getQuaternionFromEuler([0,0,pi]))
        for i in range(100):
            action[0,0] += 0.005
            self.solve_all_robots(robot_frame, action)
            time.sleep(1)