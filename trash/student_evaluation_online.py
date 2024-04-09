import os
import numpy as np
from tqdm import tqdm, trange
import yaml
import argparse

from rialto.algo.buffer_distillation import OnlineBuffer, OnlineBufferHistory
import copy

import numpy as np
from tqdm import tqdm

import torch
from torch.cuda import amp
import wandb

from utils import preprocess_points_feats

from rialto.algo.spcnn import SparseConvPolicy, SparseRNNConvPolicy
from omni.isaac.kit import SimulationApp

import time

from utils import rollout_policy, visualize_trajectory, create_env
import os

def collect_rollout(num_rollouts, env, student_policy, teacher_policy, cfg):
    num_traj = 0
    if cfg["rnn"]:
        buffer = OnlineBufferHistory(cfg=cfg)
    else:
        buffer = OnlineBuffer(cfg=cfg)
    if cfg["from_disk"]:
        num_demos = cfg["num_demos"]
        traj_idxs = np.random.choice(num_demos, num_rollouts)
        folder_name = os.path.join(cfg["main_folder"],cfg["filename"])

        actions = np.load(folder_name+"/demo_actions.npy")[traj_idxs]
        joints = np.load(folder_name+"/demo_joints.npy")[traj_idxs]
        cont_actions = np.load(folder_name+"/demo_cont_actions.npy")[traj_idxs]
        states = np.load(folder_name+"/demo_states.npy")[traj_idxs]
        all_pcd_colors = []
        all_pcd_points = []
        
        for traj_idx in traj_idxs:
            print("trajidx", traj_idx)
            traj_name = f"/traj_{traj_idx}_points.npy"
            points = np.load(folder_name+traj_name)
            feats = np.ones_like(points)
            all_pcd_points.append(points)
            all_pcd_colors.append(feats)
        all_pcd_points = np.array(all_pcd_points)
        all_pcd_colors = np.array(all_pcd_colors)

        buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors)
    else: 
        while num_traj < num_rollouts:
            if cfg["dagger"]:
                actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, student_policy, buffer.urdf, cfg, render=True, from_state=False, expert_policy=teacher_policy)

                buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors, expert_actions)
            
                num_traj += np.sum(actions.shape[0])

                cfg["current_demos"] += np.sum(actions.shape[0])

            else:
                actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, teacher_policy, buffer.urdf, cfg, render=True, from_state=True)
            
                buffer.add_trajectories(actions[success], cont_actions[success], states[success], joints[success], all_pcd_points[success], all_pcd_colors[success], None)

                num_traj += np.sum(success)
                cfg["current_demos"] += np.sum(success)


            visualize_trajectory(images, success)

            
            wandb.log({"teacher_success": np.mean(success), "num_rollouts":cfg["current_demos"]})

        buffer.store(cfg["current_demos"], cfg["filename"])

    return buffer


def train_policy(cfg, env, student_policy, teacher_policy, device, run_path, amp_enabled=True):
    lr = 1e-3
    if cfg['from_state']:
        lr = 0.0005

    policy_optimizer = torch.optim.Adam(
        student_policy.parameters(),
        lr
    )

    scaler = amp.GradScaler(enabled=amp_enabled)
    i = 0
    train_step = 0
    for step in tqdm(
        range(cfg['policy_eval_steps']), desc="Policy training"
    ):
        _, _, _, _, _, _, _, _, images, _, success = rollout_policy(env, student_policy, OnlineBuffer().urdf, cfg, from_state=False)

        visualize_trajectory(images, success, "eval")
        wandb.log({"EvalSuccess": np.mean(success), "eval_step": i})
        i += 1

    torch.save(student_policy.state_dict(), os.path.join("checkpoints/"+run_path, f"policy_distill.pt"))
    wandb.save(os.path.join("checkpoints/"+run_path, f"policy_distill.pt"))

    return student_policy

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_experiment(cfg):
    run_path = wandb.init(project="student_evaluation", config=cfg)
    run_path = run_path.path
    os.makedirs(f"checkpoints/{run_path}")
    set_random_seed(cfg['seed'])

    device = "cuda" 
    cfg["current_demos"] = 0
   
    env, teacher_policy = create_env(cfg, cfg['display'])
    # Train policy
    teacher_policy = teacher_policy.to(device)

    if cfg["rnn"]:
        policy_distill = SparseRNNConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            obs_size=env.observation_space.shape[0],
            act_size=cfg['act_dim'],
            hidden_size=cfg['hidden_size'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            layers=cfg['layers'],
        )
    else:
        policy_distill = SparseConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            obs_size=env.observation_space.shape[0],
            act_size=cfg['act_dim'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            layers=cfg['layers'],
        )

    if "run_path_student" in cfg:
        student_model_name = cfg["model_name_student"]

        expert_policy = wandb.restore(f"checkpoints/{student_model_name}.pt", run_path=cfg["run_path_student"])
        policy_distill.load_state_dict(torch.load(expert_policy.name, map_location=device))
    
    policy_distill = train_policy(cfg, env, policy_distill, teacher_policy, device, run_path)



MAIN_FOLDER = "/scratch/marcel/data/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd_name",type=str, default=None)
    parser.add_argument("--env_name",type=str, default="isaac-env")
    parser.add_argument("--model_name",type=str, default=None)
    parser.add_argument("--model_name_student",type=str, default=None)
    parser.add_argument("--run_path",type=str, default=None)
    parser.add_argument("--run_path_student",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--policy_batch_size", type=int, default=None)
    parser.add_argument("--dagger", action="store_true", default=False)
    parser.add_argument("--from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--random_augmentation", action="store_true", default=False)
    parser.add_argument("--std_noise", type=float, default=0.0)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--max_path_length", type=int, default=25)
    parser.add_argument("--policy_eval_steps", type=int, default=2000)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--render_images", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=7)
    parser.add_argument("--num_trajs_per_step", type=int, default=50)
    parser.add_argument("--policy_bc_epochs", type=int, default=1)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--num_cameras", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--filename",type=str, default="trash")
    parser.add_argument("--main_folder",type=str, default="/scratch/marcel/data/")
    parser.add_argument("--from_disk", action="store_true", default=False)
    parser.add_argument("--num_demos", type=int, default=100)
    parser.add_argument("--sampling_expert", type=float, default=0.0)
    parser.add_argument("--rnn", action="store_true", default=False)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    cfg = config["common"]

    cfg.update(config[args.env_name])
    cfg.update(config["teacher_student_distillation"])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value

    cfg.update(config["teacher_student_distillation"])
    cfg.update({'num_envs': args.num_envs})



    if args.extra_params is not None:
        cfg.update(config[args.extra_params])

    if args.rnn:
        cfg.update(config["rnn"])

    
    # cfg["randomize_action_mag"] = np.array([0,0,0,0,0,0])

    run_experiment(cfg)
