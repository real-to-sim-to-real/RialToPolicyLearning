import os.path as osp
import numpy as np
import torch
import cvxpy as cp
from scipy.spatial.transform import Rotation as R

from polymetis import GripperInterface, RobotInterface

from rdt.common import util, path_util
from rdt.polymetis_robot_utils.traj_util import PolymetisTrajectoryUtil


class FrankaQPIK:
    def __init__(self, robot, traj_helper):
        self.robot = robot
        self.traj_helper = traj_helper

        self.robot_model = self.robot.robot_model
    
    def diffik_traj(self, ee_pose_des_traj, total_time=5.0, precompute=True, execute=False, 
                    start_ee_pose_mat=None, start_joint_pos=None, use_pinv=False):

        if start_ee_pose_mat is None:
            # get current ee pose
            current_ee_pose = self.robot.get_ee_pose()
            current_ee_pose_mat = self.traj_helper.polypose2mat(current_ee_pose)
        else:
            current_ee_pose_mat = start_ee_pose_mat

        # using desired ee pose, compute desired ee velocity
        ee_pose_mat_traj = np.asarray([current_ee_pose_mat] + ee_pose_des_traj).reshape(-1, 4, 4)
        ee_pos_traj = ee_pose_mat_traj[:, :-1, -1].reshape(-1, 3)
        dt = total_time / ee_pos_traj.shape[0]

        # get orientations represented as axis angles, and separate into angles and unit-length axes
        ee_rot_traj = R.from_matrix(ee_pose_mat_traj[:, :-1, :-1])

        # get translation velocities with finite difference
        ee_trans_vel_traj = (ee_pos_traj[1:] - ee_pos_traj[:-1]) / dt
        ee_trans_vel_traj = np.concatenate((ee_trans_vel_traj, np.array([[0.0, 0.0, 0.0]])), axis=0)

        ee_rot_traj_inv = ee_rot_traj.inv()
        ee_delta_rot_traj = (ee_rot_traj[1:] * ee_rot_traj_inv[:-1])
        ee_delta_axis_angle_traj = ee_delta_rot_traj.as_rotvec()
        ee_rot_vel_traj = ee_delta_axis_angle_traj / dt
        ee_rot_vel_traj = np.concatenate((ee_rot_vel_traj, np.array([[0.0, 0.0, 0.0]])), axis=0)

        # combine into trajectory of spatial velocities
        ee_des_spatial_vel_traj = np.concatenate((ee_trans_vel_traj, ee_rot_vel_traj), axis=1)
        ee_velocity_desired = torch.from_numpy(ee_des_spatial_vel_traj).float()

        if start_joint_pos is None:
            # get current configuration and compute target joint velocity
            current_joint_pos = self.robot.get_joint_positions()
            current_joint_vel = self.robot.get_joint_velocities()
            print(f'Joint velocity is {current_joint_vel}')
        else:
            print(f'Starting from joint angles: {start_joint_pos}')
            current_joint_pos = torch.Tensor(start_joint_pos)

        if precompute:
            # in a loop, compute target joint velocity, integrate to get next joint position, and repeat 
            joint_pos_traj = []
            joint_vel_traj = []
            for t in range(ee_velocity_desired.shape[0]):
                jacobian = self.robot_model.compute_jacobian(current_joint_pos) 

                if use_pinv:
                    # solve J.pinv() @ ee_vel_des
                    joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired[t]).solution
                else:
                    # solve QP
                    from IPython import embed; embed()
                    
                    v_max = 2.7
                    G = np.vstack([
                        np.eye(len(current_joint_vel)),
                        -1.0 * np.eye(len(current_joint_vel))])
                    h = np.vstack([
                        v_max * np.ones(len(current_joint_vel)*2).reshape(-1, 1)]).reshape(-1)

                    # h =  v_max * np.ones(len(current_joint_vel))
                    v = cp.Variable(len(current_joint_vel))

                    error = jacobian @ v - ee_velocity_desired[t]

                    # prob = cp.Problem(cp.Minimize((1/2) * cp.quad_form(v, P) + q.T @ v),
                    #                   [G @ v <= h,
                    #                    A @ v == b])

                    prob = cp.Problem(cp.Minimize(cp.norm(error)),
                                      [G@v <= h])
                    # prob = cp.Problem(cp.Minimize(cp.norm(error)))

                    prob.solve()
                    out = v.value

                    raise NotImplementedError

                joint_pos_desired = current_joint_pos + joint_vel_desired*dt

                joint_pos_traj.append(joint_pos_desired)
                joint_vel_traj.append(joint_vel_desired.detach().cpu().numpy())

                current_joint_pos = joint_pos_desired.clone()

            joint_pos_desired = torch.stack(joint_pos_traj, dim=0)
            joint_vel_desired = np.stack(joint_vel_traj)
            max_idx = np.argmax(np.max(joint_vel_desired, axis=1))
            print(f'Max velocity is {joint_vel_desired[max_idx]}')
            if np.max(np.absolute(joint_vel_desired[max_idx])) > 2.7:
                print(f'Over max velocity...')
                return

            if execute:
                # self.traj_helper.execute_position_path(joint_pos_desired)
                print(f'Not executing anything right now!')
                pass

            return joint_pos_desired
        else:
            if not execute:
                print(f'[Trajectory Util DiffIK] "Execute" must be True when running with precompute=False')
                print(f'[Trajectory Util DiffIK] Exiting')
                return 
            
            # in a loop, compute target joint velocity, integrate to get next joint position, and repeat 
            vel_lp_alpha = 0.1
            vel_ramp_down_alpha = 0.9
            ramp_down_coef = 1.0

            pdt_ = self._min_jerk_spaces(ee_pose_mat_traj.shape[0], total_time)[1]
            pdt = pdt_ / pdt_.max()

            joint_pos_traj = []
            # for t in range(ee_pose_mat_traj.shape[0]):
            for t_idx in range(ee_pose_mat_traj.shape[0]):
                
                t = t_idx + self.diffik_lookahead
                if t >= (ee_pose_mat_traj.shape[0] - 1):
                    t = ee_pose_mat_traj.shape[0] - 1

                # compute velocity needed to get to next desired pose, from current pose
                current_ee_pose = self.robot.get_ee_pose()
                current_ee_pose_mat = self.traj_helper.polypose2mat(current_ee_pose)

                # get current
                current_ee_pos = current_ee_pose_mat[:-1, -1].reshape(1, 3)
                current_ee_ori_mat = current_ee_pose_mat[:-1, :-1]
                current_ee_rot = R.from_matrix(current_ee_ori_mat)

                # get desired
                ee_pos_des = ee_pos_traj[t].reshape(1, 3)
                ee_rot_des = ee_rot_traj[t]

                # compute desired rot as delta_rot, in form of axis angle
                delta_rot = (ee_rot_des * current_ee_rot.inv())
                delta_axis_angle = delta_rot.as_rotvec().reshape(1, 3)

                # stack into desired spatial vel
                trans_vel_des = (ee_pos_des - current_ee_pos) / dt
                rot_vel_des = (delta_axis_angle) / dt
                ee_velocity_desired = torch.from_numpy(
                    np.concatenate((trans_vel_des, rot_vel_des), axis=1).squeeze()).float()

                # solve J.pinv() @ ee_vel_des
                jacobian = self.robot_model.compute_jacobian(current_joint_pos) 
                joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
                joint_pos_desired = current_joint_pos + joint_vel_desired*dt

                # send joint angle command
                # self.robot.update_desired_joint_positions(joint_pos_desired)
                print(f'Not executing anything right now!')

                current_joint_pos = self.robot.get_joint_positions()
                current_joint_vel = self.robot.get_joint_velocities()

            return None
    

if __name__ == "__main__":

    import argparse
    import meshcat
    parser = argparse.ArgumentParser()
    parser.add_argument('--port_vis', '-p', type=int, default=6001)

    args = parser.parse_args()

    zmq_url=f'tcp://127.0.0.1:{args.port_vis}'
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    franka_ip = "173.16.0.1" 
    robot = RobotInterface(ip_address=franka_ip)

    traj_helper = PolymetisTrajectoryUtil(robot=robot)
    n_med = 500
    traj_helper.set_diffik_lookahead(int(n_med * 7.0 / 100))

    qp_ik = FrankaQPIK(robot, traj_helper)

    current_pose = robot.get_ee_pose()

    from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
    poly_util = PolymetisHelper()
    current_pose_mat = poly_util.polypose2mat(current_pose)

    from rdt.polymetis_robot_utils.plan_exec_util import PlanningHelper
    gripper = None  # GripperInterface(ip_address=franka_ip)
    tmp_obstacle_dir = osp.join(path_util.get_rdt_obj_descriptions(), 'tmp_planning_obs')

    from rdt.common.franka_ik import FrankaIK
    ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], no_gripper=True, mc_vis=mc_vis)

    planning = PlanningHelper(
        mc_vis=mc_vis,
        robot=robot,
        gripper=gripper,
        ik_helper=ik_helper,
        traj_helper=traj_helper,
        tmp_obstacle_dir=tmp_obstacle_dir
    )

    new_pose_mat = current_pose_mat.copy()
    new_pose_mat[:-1, -1] += np.array([0.05, 0.0, 0.0])

    to_new_pose_mats = planning.interpolate_pose(current_pose_mat, new_pose_mat, 500)
    for i in range(len(to_new_pose_mats)):
        if i % 50 == 0:
            util.meshcat_frame_show(mc_vis, f'scene/poses/to_next/{i}', to_new_pose_mats[i])


    print("here with qp_ik")
    from IPython import embed; embed()

    move_time = 3.0
    joint_traj_diffik = qp_ik.diffik_traj(
        to_new_pose_mats,
        precompute=True,
        execute=False,
        total_time=move_time)


    print("here with joint_traj_diffik")
    from IPython import embed; embed()

    joint_traj_diffik_list = [val.tolist() for val in joint_traj_diffik]
    log_info(f'Checking feasibility of DiffIK traj...')
    planning.execute_pb_loop(joint_traj_diffik_list[::10])

    valid_ik = True 
    for jnt_val in joint_traj_diffik_list:
        ik_helper.set_jpos(jnt_val)
        in_collision = ik_helper.check_collision()
        valid_ik = valid_ik and in_collision
        if not valid_ik:
            break