# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import Dict, Optional, Tuple

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_inv,
)
import cvxpy as cp
import numpy as np
from scipy.spatial.transform import Rotation as R


@configclass
class DifferentialInverseKinematicsCfg:
    """Configuration for inverse differential kinematics controller."""

    command_type: str = MISSING
    """Type of command: "position_abs", "position_rel", "pose_abs", "pose_rel"."""

    ik_method: str = MISSING
    """Method for computing inverse of Jacobian: "pinv", "svd", "trans", "dls"."""

    ik_params: Optional[Dict[str, float]] = None
    """Parameters for the inverse-kinematics method. (default: obj:`None`).

    - Moore-Penrose pseudo-inverse ("pinv"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
    - Adaptive Singular Value Decomposition ("svd"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
        - "min_singular_value": Single values less than this are suppressed to zero (default: 1e-5).
    - Jacobian transpose ("trans"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
    - Damped Moore-Penrose pseudo-inverse ("dls"):
        - "lambda_val": Damping coefficient (default: 0.1).
    """

    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from parent body to end-effector frame in parent body frame."""
    rotation_offset: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotational offset from parent body to end-effector frame in parent body frame."""

    position_command_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the position command received. Used only in relative mode."""
    rotation_command_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the rotation command received. Used only in relative mode."""


class QPDifferentialInverseKinematics:
    """Inverse differential kinematics controller.

    This controller uses the Jacobian mapping from joint-space velocities to end-effector velocities
    to compute the delta-change in the joint-space that moves the robot closer to a desired end-effector
    position.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:
        - "pinv": Moore-Penrose pseudo-inverse
        - "svd": Adaptive singular-value decomposition (SVD)
        - "trans": Transpose of matrix
        - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)

    Note: We use the quaternions in the convention: [w, x, y, z].

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
        [2] https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    """

    _DEFAULT_IK_PARAMS = {
        "pinv": {"k_val": 1.0},
        "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
        "trans": {"k_val": 1.0},
        "dls": {"lambda_val": 0.1},
    }
    """Default parameters for different inverse kinematics approaches."""

    def __init__(self, cfg: DifferentialInverseKinematicsCfg, num_robots: int, device: str):
        """Initialize the controller.

        Args:
            cfg (DifferentialInverseKinematicsCfg): The configuration for the controller.
            num_robots (int): The number of robots to control.
            device (str): The device to use for computations.

        Raises:
            ValueError: When configured IK-method is not supported.
            ValueError: When configured command type is not supported.
        """
        # store inputs
        self.cfg = cfg
        self.num_robots = num_robots
        self._device = device
        # check valid input
        if self.cfg.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}.")
        if self.cfg.command_type not in ["position_abs", "position_rel", "pose_abs", "pose_rel"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.cfg.command_type}.")

        # update parameters for IK-method
        self._ik_params = self._DEFAULT_IK_PARAMS[self.cfg.ik_method].copy()
        if self.cfg.ik_params is not None:
            self._ik_params.update(self.cfg.ik_params)
        # end-effector offsets
        # -- position
        tool_child_link_pos = torch.tensor(self.cfg.position_offset, device=self._device)
        self._tool_child_link_pos = tool_child_link_pos.repeat(self.num_robots, 1)
        # -- orientation
        tool_child_link_rot = torch.tensor(self.cfg.rotation_offset, device=self._device)
        self._tool_child_link_rot = tool_child_link_rot.repeat(self.num_robots, 1)
        # transform from tool -> parent frame
        self._tool_parent_link_rot = quat_inv(self._tool_child_link_rot)
        self._tool_parent_link_pos = -quat_apply(self._tool_parent_link_rot, self._tool_child_link_pos)
        # scaling of command
        self._position_command_scale = torch.diag(torch.tensor(self.cfg.position_command_scale, device=self._device))
        self._rotation_command_scale = torch.diag(torch.tensor(self.cfg.rotation_command_scale, device=self._device))

        # create buffers
        self.desired_ee_pos = torch.zeros(self.num_robots, 3, device=self._device)
        self.desired_ee_rot = torch.zeros(self.num_robots, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        if "position" in self.cfg.command_type:
            return 3
        elif self.cfg.command_type == "pose_rel":
            return 6
        elif self.cfg.command_type == "pose_abs":
            return 7
        else:
            raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

    """
    Operations.
    """

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        # check input size
        if command.shape != (self.num_robots, self.num_actions):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_robots, self.num_actions)}'."
            )
        # store command
        self._command[:] = command


    def diffik_traj(self, ee_pose_des_traj,start_ee_pose_mat, dt=0.5, precompute=True, execute=False, 
                    start_joint_pos=None, use_pinv=False):

        import IPython
        IPython.embed()
        # get current ee pose
        current_ee_pose_mat = start_ee_pose_mat

        # using desired ee pose, compute desired ee velocity
        ee_pose_mat_traj = np.asarray([current_ee_pose_mat] + ee_pose_des_traj).reshape(-1, 4, 4)
        ee_pos_traj = ee_pose_mat_traj[:, :-1, -1].reshape(-1, 3)
        # dt = total_time / ee_pos_traj.shape[0]

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
        

    def compute(
        self,
        current_ee_pos: torch.Tensor,
        current_ee_rot: torch.Tensor,
        jacobian: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Returns:
            torch.Tensor: The target joint positions commands.
        """
        # compute the desired end-effector pose
        if "position_rel" in self.cfg.command_type:
            # scale command
            self._command @= self._position_command_scale
            # compute targets
            self.desired_ee_pos = current_ee_pos + self._command
            self.desired_ee_rot = current_ee_rot
        elif "position_abs" in self.cfg.command_type:
            # compute targets
            self.desired_ee_pos = self._command
            self.desired_ee_rot = current_ee_rot
        elif "pose_rel" in self.cfg.command_type:
            # scale command
            self._command[:, 0:3] @= self._position_command_scale
            self._command[:, 3:6] @= self._rotation_command_scale
            # compute targets
            self.desired_ee_pos, self.desired_ee_rot = apply_delta_pose(current_ee_pos, current_ee_rot, self._command)
        elif "pose_abs" in self.cfg.command_type:
            # compute targets
            self.desired_ee_pos = self._command[:, 0:3]
            self.desired_ee_rot = self._command[:, 3:7]
        else:
            raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

        # transform from ee -> parent
        # TODO: Make this optional to reduce overhead?
        desired_parent_pos, desired_parent_rot = combine_frame_transforms(
            self.desired_ee_pos, self.desired_ee_rot, self._tool_parent_link_pos, self._tool_parent_link_rot
        )
        # transform from ee -> parent
        # TODO: Make this optional to reduce overhead?
        current_parent_pos, current_parent_rot = combine_frame_transforms(
            current_ee_pos, current_ee_rot, self._tool_parent_link_pos, self._tool_parent_link_rot
        )
        # compute pose error between current and desired
        position_error, axis_angle_error = compute_pose_error(
            current_parent_pos, current_parent_rot, desired_parent_pos, desired_parent_rot, rot_error_type="axis_angle"
        )
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_positions = self._compute_delta_dof_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_positions = self._compute_delta_dof_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_positions + delta_joint_positions

    """
    Helper functions.
    """

    def _compute_delta_dof_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in dos-position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired end-effector
        position.

        Args:
            delta_pose (torch.Tensor): The desired delta pose in shape [N, 3 or 6].
            jacobian (torch.Tensor): The geometric jacobian matrix in shape [N, 3 or 6, num-dof]

        Returns:
            torch.Tensor: The desired delta in joint space.
        """
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self._ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self._ik_params["k_val"]
            min_singular_value = self._ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-dof
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self._ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self._ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_dof_pos
