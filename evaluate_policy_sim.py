import argparse
import wandb
import numpy as np
import yaml
import os
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
from utils import create_panda_urdf, create_pcd_policy, visualize_trajectory, create_env, rollout_policy


def run(run_path=None,
        num_episodes=2,
        cam_index=2,
        **cfg
        ):

    urdf = create_panda_urdf(cfg)
    urdf.build_canon_robot_pcds(n_pts_per_link=int(cfg['arm_num_points'] / len(urdf.cfg)))  # dict of point clouds - keys are link names, values are np.ndarrays (Nx3) in canonical pose, ready to be transformed by world link pose)

    env, _ = create_env(cfg, cfg["display"])
    
    if cfg["end_model"] == -1:
        cfg["end_model"] = 1000
    cfg["end_model"] = max(cfg["start_model"]+1, cfg["end_model"])


    for num in range(cfg["start_model"], cfg["end_model"], cfg["step_model"]):
        policy = create_pcd_policy(cfg, env, "policy_distill_step_"+str(num), run_path)
        all_success = 0
        all_runs = 0
        for i in range(num_episodes):
            actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, policy, urdf, cfg, from_state=False, render=True, visualize_traj=True)

            visualize_trajectory(images, success)

            all_success += np.sum(success)
            all_runs += success.shape[0]

        wandb.log({"model_num":num, "EvalSuccess": all_success/all_runs})


def env_distance(env, state, goal):
        obs = env.observation(state)
        
        return env.compute_shaped_distance(obs, goal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name",type=str, default=None)
    parser.add_argument("--env_name",type=str, default="isaac-env")
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=12)
    # parser.add_argument("--crop_min", type=float, default=0)
    parser.add_argument("--hz", type=float, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--start_model", type=int, default=0)
    parser.add_argument("--end_model", type=int, default=-1)
    parser.add_argument("--step_model", type=int, default=1)
    parser.add_argument("--extra_params",type=str, default=None)
    parser.add_argument("--rnn", action="store_true", default=False)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")
    parser.add_argument("--layers",type=str, default=None)
    parser.add_argument("--pool",type=str, default=None)
    parser.add_argument("--voxel_size",type=float, default=None)
    parser.add_argument("--pcd_encoder_type",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--datafolder",type=str, default=None)
    parser.add_argument("--distractors",type=str, default="no_distractors")
    parser.add_argument("--use_synthetic_pcd", action="store_true", default=False)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--decimation", type=int, default=None)

    args = parser.parse_args()
    params = {}

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]

    params.update(config["isaac-env"])

    data_folder_name = f"isaac_env_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name

    params["data_folder"] = data_folder_name


    params.update(config['teacher_student_distillation'])
    params.update(config[args.pcd_randomness])
    params.update(config[args.distractors])

    params.update({"max_path_length":args.max_path_length})
    # params.update({"use_synthetic_pcd":False})
    

    
    if args.voxel_size is not None:
        params.update({"voxel_size": args.voxel_size})
    
    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            params.update(config[extra_param])
    
    if args.use_synthetic_pcd:
        params.update(config["synthetic_pcd"])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    # params.update({"display": True,
    #                 "num_envs": 1,
    #                 "render_images": True,
    #                })

    wandb.init(project="sim_franka_"+params["usd_name"]+"_evaluation", name=f"real_franka_eval", config=params, dir="/data/pulkitag/data/marcel/wandb")

    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
