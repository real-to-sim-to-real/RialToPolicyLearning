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

try:
    import torchsparse
    have_sparse = True
except ImportError as e:
    have_sparse = False
if have_sparse:
    from rialto.algo.spcnn import SparseConvPolicy, SparseRNNConvPolicy

from omni.isaac.kit import SimulationApp

import time

from utils import rollout_policy, visualize_trajectory, create_env, create_state_policy, create_pcd_policy
import os
from os import listdir
from os.path import isfile, join
import gc

all_trajs_files = None
all_trajs_files_demo_num = None
buffer = None

def collect_rollout(num_rollouts, env, student_policy, teacher_policy, from_disk, cfg):
    num_traj = 0
    global buffer 

    if buffer is not None and cfg["reuse_buffer"]:
        return buffer
    
    if cfg["rnn"]:
        buffer = OnlineBufferHistory(cfg=cfg)
    else:
        buffer = OnlineBuffer(cfg=cfg)

    if from_disk:
        filename = cfg["filename"]
        if "datafolder" in cfg:
            datafolder = cfg["datafolder"]
        else:
            datafolder = "/data/pulkitag/results/marcel/data/"
        folder_name = f"{datafolder}/{filename}" #os.path.join(cfg["main_folder"],cfg["filename"])

        global all_trajs_files
        global all_trajs_files_demo_num
        if all_trajs_files is None:
            onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
            all_trajs_files = []
            all_trajs_files_demo_num = []
            for file in onlyfiles:
                if "actions" in file:
                    postfix = file[8:]
                    # num = int(file.split("_")[-1].split(".")[0])
                    node = postfix.split("_")[0]
                    all_trajs_files.append(postfix)
                    all_trajs_files_demo_num.append(int(postfix.split("_")[1].split(".")[0]))

            sorted_idxs = np.argsort(all_trajs_files_demo_num)
            all_trajs_files = np.array(all_trajs_files)[sorted_idxs]

        num_traj = 0
        while num_traj < num_rollouts:
            choice = len(all_trajs_files)
            if "max_demos" in cfg and cfg["max_demos"] > 0:
                choice = min(len(all_trajs_files), cfg["max_demos"])

            idx = np.random.choice(choice, 1)[0]

            traj_postfix = all_trajs_files[idx ]
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

            if not cfg['student_from_state']:
                all_pcd_points = np.load(folder_name+f"/pcd_points_{traj_postfix}")
                all_pcd_colors = None #np.ones_like(all_pcd_points)
            else:
                all_pcd_points = None
                all_pcd_colors = None

            num_traj += actions.shape[0]

            buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors)

    else: 
        while num_traj < num_rollouts:
            render = not cfg["teacher_from_state"] or cfg["render_images"]
            if cfg["dagger"]:
                actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, student_policy, buffer.urdf, cfg, render=render, from_state=cfg["teacher_from_state"], expert_policy=teacher_policy, visualize_traj=render)

                buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors, expert_actions)
            
                num_traj += np.sum(actions.shape[0])

                cfg["current_demos"] += np.sum(actions.shape[0])

            else:
                actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, teacher_policy, buffer.urdf, cfg, render=render, from_state=cfg["teacher_from_state"], visualize_traj=render)

                buffer.add_trajectories(actions[success], cont_actions[success], states[success], joints[success], all_pcd_points[success], all_pcd_colors[success], None)

                num_traj += np.sum(success)
                cfg["current_demos"] += np.sum(success)

            if cfg["visualize_traj"]:
                visualize_trajectory(images, success)

            
            wandb.log({"teacher_success": np.mean(success), "num_rollouts":cfg["current_demos"]})
        
        if cfg["store_traj"]:
            buffer.store(cfg["current_demos"], cfg["filename"], cfg["node"], cfg["datafolder"])

    return buffer


def train_policy(cfg, env, student_policy, teacher_policy, device, run_path, amp_enabled=True):
    lr = cfg['lr']
    # if cfg['student_from_state']:
    #     lr = 0.0005

    weight_decay = 0
    if cfg["weight_decay"]:
        weight_decay = cfg["weight_decay"]

    policy_optimizer = torch.optim.Adam(
        student_policy.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    scaler = amp.GradScaler(enabled=amp_enabled)
    i = 0
    train_step = 0
    print("Num trajs per step", cfg["num_trajs_per_step"])
    for step in tqdm(
        range(cfg['policy_train_steps']), desc="Policy training"
    ):

        rollout_data = collect_rollout(cfg['num_trajs_per_step'], env, student_policy, teacher_policy, cfg["from_disk"], cfg)
        # with torch.no_grad():
        #     actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, student_policy, rollout_data.urdf, cfg, from_state=cfg["student_from_state"], render=True, visualize_traj=cfg['visualize_traj'])

        if not cfg["only_collect"]:
            val_traj_idxs_batches = None
            for epoch in tqdm(
                range(cfg['policy_bc_epochs']), desc="Policy training"
            ):
                student_policy.train()
                traj_idxs_batches, val_traj_idxs_batches = rollout_data.sample_idxs(cfg['policy_batch_size'], val_traj_idxs_batches)
                start = time.time()
                for idx, traj_idxs in enumerate(traj_idxs_batches):
                    start_batch = time.time()
                    train_step+=1
                    # Sample batch
                    points, feats, state, joint, act, expert_act, full_state = rollout_data.sample(
                        traj_idxs #, cfg.env.max_episode_steps
                    )
                    print("Sample data took", time.time() - start_batch)

                    state = torch.as_tensor(state, dtype=torch.float32).to(device)

                    if cfg["dagger"]:
                        act = torch.as_tensor(expert_act, dtype=torch.long).to(device)
                    else:
                        act = torch.as_tensor(act, dtype=torch.long).to(device)

                    # Compute policy loss
                    with amp.autocast(enabled=True):

                        if cfg['student_from_state']:
                            state = torch.as_tensor(full_state, dtype=torch.float32).to(device)

                            logits = student_policy(state, state)
                            # import torch
                            policy_loss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0)(logits, act)
                            policy_loss = torch.mean(policy_loss)
                            acc = torch.mean((logits.argmax(dim=1) == act).float()).item()

                        else:
                            policy_loss = student_policy.compute_loss(points, feats, state, act)
                            
                            # logits = student_policy((points,feats,state))
                            # acc = torch.mean((logits.argmax(dim=1) == act).float()).item()
                            acc = -1

                    policy_optimizer.zero_grad()
                    scaler.scale(policy_loss).backward()
                    scaler.step(policy_optimizer)
                    scaler.update()
                    policy_loss = policy_loss.item()

                    print("policy loss", policy_loss, time.time() - start_batch)
                    
                    wandb.log({"train/loss":policy_loss, "train_step":train_step, "train acc":acc})


                with torch.no_grad():
                    val_losses = []
                    for val_traj_idxs in val_traj_idxs_batches:
                        val_points, val_feats, val_state, val_joint, val_act, val_expert_act, val_full_state = rollout_data.sample(
                            val_traj_idxs
                        )
                        val_state = torch.as_tensor(val_state, dtype=torch.float32).to(device)

                        if cfg["dagger"]:
                            val_act = torch.as_tensor(val_expert_act, dtype=torch.long).to(device)
                        else:
                            val_act = torch.as_tensor(val_act, dtype=torch.long).to(device)
                    
                        # Compute policy loss
                        with amp.autocast(enabled=True):
                            if cfg['student_from_state']:

                                val_state = torch.as_tensor(val_full_state, dtype=torch.float32).to(device)

                                logits = student_policy(val_state, val_state)
                                # import torch
                                val_loss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0)(logits, val_act)
                                val_loss = torch.mean(val_loss)
                                acc = torch.mean((logits.argmax(dim=1) == val_act).float()).item()

                            else:
                                val_loss = student_policy.compute_loss(val_points, val_feats, val_state, val_act)

                                # logits = student_policy((val_points,val_feats,val_state))
                                # acc = torch.mean((logits.argmax(dim=1) == val_act).float()).item()
                                acc = -1

                        val_loss = val_loss.item()
                        val_losses.append(val_loss)
                        print("val loss", np.mean(val_losses))

                print("Epoch step:", epoch, time.time() - start)
                
                wandb.log({"val/loss":val_loss, "val/acc":acc})

            wandb.log({"step":step})
            

            if cfg["eval_freq"]!=0 and (step + 1) % cfg["eval_freq"] == 0:
                n = 0
                all_success = 0
                all_trials = 0
                torch.save(
                    student_policy.state_dict(),
                    os.path.join(
                        cfg["datafolder"], f"policy_distill.pt"
                    ),
                )

                torch.save(
                    student_policy.state_dict(),
                    os.path.join(
                        "checkpoints", f"policy_distill_step_{i}.pt"
                    ),
                )

                wandb.save(os.path.join(
                        "checkpoints", f"policy_distill_step_{i}.pt"
                    ))
                
                while n < cfg["num_trajs_eval"]:
                    render = not cfg["student_from_state"] or cfg["render_images"]
                    with torch.no_grad():
                        actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success = rollout_policy(env, student_policy, rollout_data.urdf, cfg, from_state=cfg["student_from_state"], render=render, visualize_traj=cfg['visualize_traj'])
                    
                    
                    # import IPython
                    # IPython.embed()     

                    # import open3d as o3d
                    # # idx = 18
                    # # pcd_demos = np.load(f"/home/marcel/SimToRealFranka/demos/realworldfranka/pcd_points_0_{idx}.npy")
                    # pcd = o3d.geometry.PointCloud()
                    # pcd_dem = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(all_pcd_points[0][0])
                    # pcd_dem.points = o3d.utility.Vector3dVector(rollout_data.pcd_points[0][0])
                    # pcd_dem.colors = o3d.utility.Vector3dVector(np.zeros_like(rollout_data.pcd_points[0][0]))
                    # o3d.visualization.draw_geometries([pcd, pcd_dem])

                    n += success.shape[0]
                    all_success += np.sum(success)
                    all_trials += len(success)
                    if cfg["eval_freq"]!=0 and (step + 1) % cfg["eval_freq"]*5 == 0:
                        print("Inside eval freq sub")
                        start = time.time()
                        if render and cfg['visualize_traj']:
                            visualize_trajectory(images, success, "eval")
                        # wandb.log({"EvalSuccess": np.mean(success), "eval_step": i})
                        print("Logging to wandb took", time.time() - start)
                wandb.log({"EvalSuccess": all_success / all_trials, "eval_step": i})
                    
                i += 1

        else:
            del rollout_data
            gc.collect()
            
    return student_policy

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_experiment(cfg):
    project_name = "distillation_online"
    # if cfg["student_from_state"] and cfg["from_disk"]:
    project_name = "distillation" + "_student_from_state"
    if cfg["only_collect"]:
        project_name = "distillation_online" + "_collect"
    project_name = project_name + f"_{cfg['env_type']}"

    if "WANDB_DIR" in os.environ:
        # use environment variable if possible
        wandb_dir = os.environ["WANDB_DIR"]
    else:
        # otherwise use argparse
        wandb_dir = cfg["wandb_dir"]
    run_path = wandb.init(project=project_name, config=cfg, dir=wandb_dir)

    run_path = run_path.path
    os.makedirs(f"checkpoints/{run_path}")
    
    set_random_seed(cfg['seed'])

    device = "cuda" 
    cfg["current_demos"] = 0
   
    if cfg['eval_freq'] == 0:
        cfg["render_images"] = False
        cfg["num_cameras"] = 0

    env = None
    if cfg["eval_freq"] != 0:
        env, _ = create_env(cfg, cfg['display'], seed=cfg['seed'])

    # Train policy
    teacher_model_name = None
    teacher_run_path = None
    if "model_name" in cfg:
        teacher_model_name = cfg["model_name"]
        teacher_run_path = cfg["run_path"]

    if cfg["from_disk"]:
        teacher_policy = None
    else:
        if cfg["teacher_from_state"]:
            teacher_policy = create_state_policy(cfg, env, teacher_model_name, teacher_run_path)
        else:
            teacher_policy = create_pcd_policy(cfg, env, teacher_model_name, teacher_run_path)

    student_model_name = None
    student_run_path = None
    if "model_name_student" in cfg:
        student_model_name = cfg["model_name_student"]
        student_run_path = cfg["run_path_student"]

    if cfg["student_from_state"]:
        policy_distill = create_state_policy(cfg, env, student_model_name, student_run_path)
    else:
        policy_distill = create_pcd_policy(cfg, env, student_model_name, student_run_path)

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
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--policy_batch_size", type=int, default=None)
    parser.add_argument("--dagger", action="store_true", default=False)
    parser.add_argument("--student_from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--random_augmentation", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--policy_train_steps", type=int, default=None)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_trajs_per_step", type=int, default=None)
    parser.add_argument("--policy_bc_epochs", type=int, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--num_cameras", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--filename",type=str, default="trash")
    parser.add_argument("--datafolder",type=str, default=None)
    parser.add_argument("--from_disk", action="store_true", default=False)
    parser.add_argument("--sampling_expert", type=float, default=None)
    parser.add_argument("--rnn", action="store_true", default=False)
    parser.add_argument("--gru", action="store_true", default=False)
    parser.add_argument("--random_config",type=str, default=None)
    parser.add_argument("--store_traj", action="store_true", default=False)
    parser.add_argument("--only_collect", action="store_true", default=False)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_trajs_eval",  type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")
    parser.add_argument("--distractors",type=str, default="no_distractors")
    parser.add_argument("--visualize_traj", action="store_true", default=False)
    parser.add_argument("--reuse_buffer", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pcd_encoder_type",type=str, default=None)
    parser.add_argument("--layers",type=str, default=None)
    parser.add_argument("--pool",type=str, default=None)
    parser.add_argument("--voxel_size",type=float, default=None)
    parser.add_argument("--max_demos", type=int, default=None)
    parser.add_argument("--use_synthetic_pcd", action="store_true", default=False)
    parser.add_argument("--presample_arm_pcd", action="store_true", default=False)
    parser.add_argument("--model_from_disk", action="store_true", default=False)
    parser.add_argument("--sample_action", action="store_true", default=False)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    cfg = config["common"]

    cfg.update(config[args.env_name])
    cfg.update(config["teacher_student_distillation"])

    if args.pcd_randomness is not None:
        cfg.update(config[args.pcd_randomness])

    cfg.update(config[args.distractors])

    if args.use_synthetic_pcd:
        cfg.update(config["synthetic_pcd"])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value

    
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

    run_experiment(cfg)
