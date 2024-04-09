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

def collect_rollout(num_rollouts, env, student_policy, cfg):
    num_traj = 0
    if cfg["rnn"]:
        buffer = OnlineBufferHistory(cfg=cfg)
    else:
        buffer = OnlineBuffer(cfg=cfg)
    if cfg["from_disk"]:

        from os import listdir
        from os.path import isfile, join
        folder_name = "/data/pulkitag/misc/marcel/data" #os.path.join(cfg["main_folder"],cfg["filename"])

        onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        all_trajs = []
        for file in onlyfiles:
            num = int(file.split("_")[-1].split(".")[0])
            all_trajs.append(num)

        num_demos = cfg["num_demos"]

        banned = [1843, -1]
        traj_idx = -1
        while traj_idx in banned:
            idx = np.random.choice(len(all_trajs), 1)[0]
            traj_idx = all_trajs[idx]
        
        # for traj_idx in traj_idxs:
        actions = np.load(folder_name+f"/actions_{traj_idx}.npy")
        joints = np.load(folder_name+f"/joints_{traj_idx}.npy")
        cont_actions = np.ones_like(actions)
        states = np.load(folder_name+f"/actions_{traj_idx}.npy")

        print("trajidx", traj_idx)
        all_pcd_points = np.load(folder_name+f"/pcd_points_{traj_idx}.npy")
        all_pcd_colors = np.ones_like(all_pcd_points)
        # all_pcd_points.append(points)
        # all_pcd_colors.append(feats)
        # all_pcd_points = np.array(all_pcd_points)
        # all_pcd_colors = np.array(all_pcd_colors)

        buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors)
   

    return buffer


def train_policy(cfg, env, student_policy, device, run_path, amp_enabled=True):
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
        range(cfg['policy_train_steps']), desc="Policy training"
    ):
        rollout_data = collect_rollout(cfg['num_trajs_per_step'], env, student_policy, cfg)

        val_traj_idxs_batches = None
        for epoch in tqdm(
            range(cfg['policy_bc_epochs']), desc="Policy training"
        ):
            student_policy.train()
            traj_idxs_batches, val_traj_idxs_batches = rollout_data.sample_idxs(cfg['policy_batch_size'], val_traj_idxs_batches)

            for idx, traj_idxs in enumerate(traj_idxs_batches):
                start = time.time()
                train_step+=1
                # Sample batch
                points, feats, state, act, expert_act = rollout_data.sample(
                    traj_idxs #, cfg.env.max_episode_steps
                )

                state = torch.as_tensor(state, dtype=torch.float32).to(device)

                if cfg["dagger"]:
                    act = torch.as_tensor(expert_act, dtype=torch.long).to(device)
                else:
                    act = torch.as_tensor(act, dtype=torch.long).to(device)

                # Compute policy loss
                with amp.autocast(enabled=True):
                    policy_loss = student_policy.compute_loss(points, feats, state, act)


                policy_optimizer.zero_grad()
                scaler.scale(policy_loss).backward()
                scaler.step(policy_optimizer)
                scaler.update()
                print("policy loss", policy_loss)
                policy_loss = policy_loss.detach().cpu()
                
                wandb.log({"train/loss":policy_loss.item(), "train_step":train_step})

            val_losses = []
            with torch.no_grad():
                for val_traj_idxs in val_traj_idxs_batches:
                    val_points, val_feats, val_state, val_act, val_expert_act = rollout_data.sample(
                        val_traj_idxs
                    )
                    val_state = torch.as_tensor(val_state, dtype=torch.float32).to(device)

                    if cfg["dagger"]:
                        val_act = torch.as_tensor(val_expert_act, dtype=torch.long).to(device)
                    else:
                        val_act = torch.as_tensor(val_act, dtype=torch.long).to(device)
                
                    # Compute policy loss
                    with amp.autocast(enabled=True):
                        if cfg['from_state']:
                            logits = student_policy(val_state, val_state)
                            # import torch
                            val_loss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0)(logits, val_act)
                            val_loss = torch.mean(val_loss)
                        else:
                            val_loss = student_policy.compute_loss(val_points, val_feats, val_state, val_act)

                    val_losses.append(val_loss.detach().cpu().item())
                    print("val loss", np.mean(val_losses))

            print("Epoch step:", epoch, time.time() - start)
            
            train_step = copy.deepcopy(cfg['policy_bc_epochs']*step + epoch)
            wandb.log({"val/loss":val_loss.item()})

        wandb.log({"step":step})
        torch.save(
            student_policy.state_dict(),
            os.path.join(
                "checkpoints", f"policy_distill_step_{step}.pt"
            ),
        )

        wandb.save(os.path.join(
                "checkpoints", f"policy_distill_step_{step}.pt"
            ))


    torch.save(student_policy.state_dict(), os.path.join("checkpoints/"+run_path, f"policy_distill.pt"))
    wandb.save(os.path.join("checkpoints/"+run_path, f"policy_distill.pt"))

    return student_policy

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_experiment(cfg):
    run_path = wandb.init(project="distillation_online_from_disk", config=cfg)
    run_path = run_path.path
    os.makedirs(f"checkpoints/{run_path}")
    set_random_seed(cfg['seed'])

    device = "cuda" 
    cfg["current_demos"] = 0
   

    if cfg["rnn"]:
        policy_distill = SparseRNNConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            obs_size=0,
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
            obs_size=0,
            act_size=cfg['act_dim'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            layers=cfg['layers'],
        )

    env = None
    policy_distill = train_policy(cfg, env, policy_distill, device, run_path)



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
