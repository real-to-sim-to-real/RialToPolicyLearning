import argparse
import wandb
import numpy as np
import yaml
import os
from rialto.franka.real_franka_env import RealFrankaEnv
# from spcnn import SparseConvPolicy
import torch
import time
import open3d as o3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import rotate_vector
from math import pi

import torch
import yourdfpy
from transforms3d.euler import euler2quat
from utils import create_panda_urdf,create_pcd_policy, visualize_trajectory
from utils_real import rollout_policy_real


def run(model_name,
        run_path,
        num_episodes=10,
        cam_index=2,
        **cfg
        ):

    urdf = create_panda_urdf(cfg)

    env = RealFrankaEnv(cam_index=cam_index, hz=cfg['hz'], background_loop=cfg["background_loop"], cfg=cfg)
    
    policy = create_pcd_policy(cfg, env, model_name, run_path)

    for i in range(num_episodes):
        actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy_real(env, policy, urdf, cfg, render=True, from_state=False, expert_policy=None)
        visualize_trajectory(images, success)
        

def env_distance(env, state, goal):
        obs = env.observation(state)
        
        return env.compute_shaped_distance(obs, goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str, default=None)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--max_path_length", type=int, default=25)
    parser.add_argument("--cam_index", type=int, default=2)
    # parser.add_argument("--crop_min", type=float, default=0)
    parser.add_argument("--hz", type=float, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--extra_params",type=str, default=None)
    parser.add_argument("--rnn", action="store_true", default=False)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")
    parser.add_argument("--layers",type=str, default=None)
    parser.add_argument("--pool",type=str, default=None)
    parser.add_argument("--voxel_size",type=float, default=None)
    parser.add_argument("--pcd_encoder_type",type=str, default=None)
    parser.add_argument("--background_loop", action="store_true", default=False)
    parser.add_argument("--reset_if_open", action="store_true", default=False)
    parser.add_argument("--gripper_force", type=float, default=None)
    parser.add_argument("--total_loop_time", type=float, default=None)
    parser.add_argument("--interp_steps", type=int, default=None)
    parser.add_argument("--start_interp_offset", type=int, default=None)
    parser.add_argument("--nb_open", type=int, default=None)

    args = parser.parse_args()
    params = {}

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]


    data_folder_name = f"isaac_env_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name

    params["data_folder"] = data_folder_name

    params.update(config["isaac-env"])
    params.update(config['teacher_student_distillation'])
    params.update(config["real_franka_params"])

    if args.pcd_randomness is not None:
        params.update(config[args.pcd_randomness])
    
    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            params.update(config[extra_param])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            if "unet" in key:
                params["unet3d"][key[5:]] = value
            else:
                params[key] = value

    params.update({"use_synthetic_pcd":False})

    wandb.init(project="real_franka"+"evaluation", name=f"real_franka_eval", config=params, dir="/data/pulkitag/data/marcel/wandb")

    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')