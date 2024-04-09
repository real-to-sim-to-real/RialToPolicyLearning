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

from utils import rollout_policy, visualize_trajectory, create_env, create_state_policy, create_pcd_policy
import os
from os import listdir
from os.path import isfile, join
import gc

all_trajs_files = None
buffer = None

def collect_rollout(num_rollouts, env, student_policy, teacher_policy, cfg):
    num_traj = 0
    global buffer 
    


    filename = cfg["filename"]
    if "datafolder" in cfg:
        datafolder = cfg["datafolder"]
    else:
        datafolder = "/data/pulkitag/results/marcel/data/"
    folder_name = f"{datafolder}/{filename}" #os.path.join(cfg["main_folder"],cfg["filename"])

    global all_trajs_files
    if all_trajs_files is None:
        onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        all_trajs_files = []
        for file in onlyfiles:
            if "actions" in file:
                postfix = file[8:]
                # num = int(file.split("_")[-1].split(".")[0])
                node = postfix.split("_")[0]
                if str(cfg["node"]) in node:
                    all_trajs_files.append(postfix)
    
    num_traj = 0
    num_file = 0
    while True:
        if cfg["rnn"]:
            buffer = OnlineBufferHistory(cfg=cfg)
        else:
            buffer = OnlineBuffer(cfg=cfg)
        idx = num_file

        traj_postfix = all_trajs_files[idx]
        # traj_postfixdebug = all_trajs_files[idx+1]

        actions = np.load(folder_name+f"/actions_{traj_postfix}")
        joints = np.load(folder_name+f"/joints_{traj_postfix}")
        cont_actions = np.ones_like(actions)

        if os.path.exists(folder_name+f"/states_{traj_postfix}"):
            states = np.load(folder_name+f"/states_{traj_postfix}")
        else:
            print("WARNING: No file states was found")
            states = joints.copy()
        
        states = states[:min(len(states), len(actions))]
        actions = actions[:min(len(states), len(actions))]

        print("trajidx", traj_postfix)

        if not cfg['student_from_state'] and not cfg["use_synthetic_pcd"]:
            all_pcd_points = np.load(folder_name+f"/pcd_points_{traj_postfix}")
            all_pcd_colors = None #np.ones_like(all_pcd_points)
        else:
            all_pcd_points = None
            all_pcd_colors = None

        num_traj += actions.shape[0]
        num_file += 1
        wandb.log({"Num trajs": num_file})
        buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors)

    
        buffer.store(cfg["current_demos"], cfg["target_filename"], cfg["node"], cfg["datafolder"])

    return buffer


def train_policy(cfg, env, student_policy, teacher_policy, device, run_path, amp_enabled=True):
    lr = cfg['lr']
    # if cfg['student_from_state']:
    #     lr = 0.0005

  
    rollout_data = collect_rollout(cfg['num_trajs_per_step'], env, student_policy, teacher_policy, cfg)

    return student_policy

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_experiment(cfg):
    project_name = "transform_synthetic"
    # if cfg["student_from_state"] and cfg["from_disk"]:
    project_name = project_name + f"_{cfg['env_type']}"
    run_path = wandb.init(project=project_name, config=cfg, dir="/data/pulkitag/results/marcel/")
    run_path = run_path.path
    os.makedirs(f"checkpoints/{run_path}")
    from omni.isaac.kit import SimulationApp
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu
    config = {"headless": not cfg["display"]}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)

    
    set_random_seed(cfg['seed'])

    device = "cuda" 
    cfg["current_demos"] = 0
   
    if cfg['eval_freq'] == 0:
        cfg["render_images"] = False
        cfg["num_cameras"] = 0

    env = None

    policy_distill = train_policy(cfg, env, None, None, device, run_path)



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
    parser.add_argument("--student_from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--random_augmentation", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--max_path_length", type=int, default=25)
    parser.add_argument("--policy_train_steps", type=int, default=2000)
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
    parser.add_argument("--datafolder",type=str, default=None)
    parser.add_argument("--from_disk", action="store_true", default=False)
    parser.add_argument("--sampling_expert", type=float, default=0.0)
    parser.add_argument("--rnn", action="store_true", default=False)
    parser.add_argument("--gru", action="store_true", default=False)
    parser.add_argument("--random_config",type=str, default=None)
    parser.add_argument("--store_traj", action="store_true", default=False)
    parser.add_argument("--only_collect", action="store_true", default=False)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trajs_eval",  type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--teacher_from_state", action="store_true", default=False)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")
    parser.add_argument("--distractors",type=str, default="no_distractors")
    parser.add_argument("--visualize_traj", action="store_true", default=False)
    parser.add_argument("--reuse_buffer", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pcd_encoder_type",type=str, default=None)
    parser.add_argument("--layers",type=str, default=None)
    parser.add_argument("--pool",type=str, default=None)
    parser.add_argument("--voxel_size",type=float, default=None)
    parser.add_argument("--max_demos", type=int, default=0)
    parser.add_argument("--use_synthetic_pcd", action="store_true", default=False)
    parser.add_argument("--target_filename",type=str, default=None)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    cfg = config["common"]

    cfg.update(config[args.env_name])
    cfg.update(config["teacher_student_distillation"])
    cfg.update(config[args.pcd_randomness])
    cfg.update(config[args.distractors])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value

    cfg.update(config["teacher_student_distillation"])
    cfg.update({'num_envs': args.num_envs})

    
    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            cfg.update(config[extra_param])
        
    if args.gru:
        cfg["rnn"] = True

    if args.rnn or args.gru:
        cfg.update(config["rnn"])

    if args.random_config is not None:
        cfg.update(config[args.random_config])

    if args.usd_name is not None:
        cfg["usd_name"] = args.usd_name
    
    if args.layers is not None:
        cfg.update({"layers": args.layers})

    if args.voxel_size is not None:
        cfg.update({"voxel_size": args.voxel_size})


    if args.policy_batch_size is not None:
        cfg.update({"policy_batch_size": args.policy_batch_size})

    cfg.update({"sensors":["synthetic_pcd"]})
    cfg.update({"use_synthetic_pcd":True})
    

    # cfg["randomize_action_mag"] = np.array([0,0,0,0,0,0])

    run_experiment(cfg)
