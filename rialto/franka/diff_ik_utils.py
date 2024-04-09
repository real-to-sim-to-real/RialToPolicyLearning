import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def polypose2mat(polypose):
    pose_mat = np.eye(4)
    pose_mat[:-1, -1] = polypose[0].numpy()
    pose_mat[:-1, :-1] = R.from_quat(polypose[1].numpy()).as_matrix()
    return pose_mat 

def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat (torch.Tensor): quaternions with real part first, as tensor of shape (..., 4).
        eps (float): The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        torch.Tensor: Rotations given as a vector in axis angle form, as a tensor
                of shape (..., 3), where the magnitude is the angle turned
                anti-clockwise in radians around the vector's direction.

    Reference:
        Based on PyTorch3D (https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554)
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        torch.abs(angle.abs()) > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def diffik_step(current_ee_pos, current_ee_quat, desired_ee_pos, desired_ee_quat, jacobian, dt=1.0):
    # get translation velocities with finite difference
    ee_trans_vel = (desired_ee_pos - current_ee_pos) / dt
    ee_trans_vel = ee_trans_vel.reshape(-1, 3)

    source_quat_inv = R.from_quat(current_ee_quat.cpu().numpy()).inv().as_matrix() #/ source_quat_norm.unsqueeze(-1)
    # q_error = q_target * q_current_inv
    quat_error = R.from_matrix(R.from_quat(desired_ee_quat.cpu().numpy()).as_matrix() @ source_quat_inv)
    axis_angle = torch.tensor(quat_error.as_rotvec()).reshape(-1, 3).to("cuda")
    ee_rot_vel = axis_angle/dt

    # combine into trajectory of spatial velocities
    ee_des_spatial_vel = torch.concatenate((ee_trans_vel, axis_angle), axis=1).float()
    # ee_velocity_desired = torch.from_numpy(ee_des_spatial_vel_traj).float()

    # jacobian = self.robot_model.compute_jacobian(current_joint_pos) 
    jacobian = jacobian.reshape(-1, 6, 7)
    joint_vel_desired = torch.linalg.lstsq(jacobian, ee_des_spatial_vel).solution
    joint_pos_delta = joint_vel_desired*dt

    return joint_pos_delta
    # self.robot.update_desired_joint_positions(joint_pos_desired)