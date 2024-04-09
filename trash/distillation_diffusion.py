import os
import hydra
import numpy as np
from tqdm import tqdm, trange
import yaml
import argparse

# from helpers.experiment import (
#     setup_wandb,
#     hydra_to_dict,
#     set_random_seed,
# )

from rialto.algo.buffer_distillation_diffusion import Buffer
# from common.utils import set_gpu_mode, get_device
# from common.logger import configure_logger, Video

import numpy as np
from tqdm import tqdm

import torch
from torch.cuda import amp
import wandb

# from common.buffers import PointsEpisodicReplayBuffer

# from environments import load_envs
from rialto.algo.spcnn import preprocess_points_feats

from rialto.algo.spcnn import SparseConvPolicy
from omni.isaac.kit import SimulationApp

import time
from distillation import create_env, create_video, set_random_seed

def evaluate_policy(env, policy, cfg):
    policy.eval()
    demo_num = 0
    num_envs = cfg['num_envs']
    max_path_length = cfg['max_path_length']
    from_state = cfg['from_state']
    use_state = cfg['use_state']
    all_success = []
    while demo_num < 10:
        print("Evaluating traj number", demo_num)
        actions = []
        states = []

        # goal = env.extract_goal(env.sample_goal())

        state = env.reset()
        joints = env.base_env._env.get_robot_joints()
        images = []
        pcds = []
        start_demo =time.time()
        sensors = ['rgb', 'pointcloud']
        if from_state:
            sensors = ['rgb']

        for t in range(max_path_length):
            start = time.time()
            if from_state:
                img = env.render_image(sensors)
            else:
                img, pcds = env.render_image(sensors)
            
                # pcd = pcd[0]
                pcd_processed_points = []
                pcd_processed_colors = []
                for pcd in pcds:
                    pcd_proc_points, pcd_proc_colors = preprocess_points_feats(np.asarray(pcd.points), np.asarray(pcd.colors), cfg['crop_min'],cfg['crop_max'],cfg['voxel_size'], rgb_feats=False,num_points=cfg['num_points'], add_padding=cfg['pad_points'], downsample_points=cfg['downsample']) #self.preprocess_pcd(pcd)

                    pcd_processed_points.append(pcd_proc_points)
                    pcd_processed_colors.append(pcd_proc_colors)
                
                pcd_processed_points = np.concatenate(pcd_processed_points)
                pcd_processed_colors = np.concatenate(pcd_processed_colors)
                
            images.append(img)
            print("Rendering pcd image", time.time()-start)
            observation = env.observation(state)

            if from_state:
                observation = torch.tensor(observation)
                action = policy(observation, observation).argmax(dim=1).cpu().numpy()
            else:
                action = policy((pcd_processed_points, pcd_processed_colors, joints)).argmax(dim=1).cpu().numpy()

            actions.append(action)
            states.append(state)

            state, _, done , info = env.step(action)
            joints = info["robot_joints"]

        success = env.base_env._env.get_success().detach().cpu().numpy()
        for s in success:
            all_success.append(s)
        images = np.array(images).transpose(1,0,2,3,4)
        if np.sum(success)>0:
            images_success = np.concatenate(images[success], axis=1)
            create_video(images_success, "success")

        if np.sum(success)< success.shape[0]:
            images_failed = np.concatenate(images[np.logical_not(success)], axis=1)
            create_video(images_failed, "failed")

        print(f"Trajectory took {time.time() - start_demo}")
        demo_num += num_envs

    return np.mean(all_success)

def train_policy(cfg, env):
    action_dim = 9 #env.action_space.shape
    obs_dim = env.observation_space.shape
    device = "cuda:0"
    obs_horizon = "hi"


    buffer = Buffer(
                folder_name=cfg['folder_name'],
                pad_points=cfg['pad_points'], 
                voxel_size=cfg['voxel_size'], 
                num_points=cfg['num_points'], 
                downsample=cfg['downsample'],
                crop_min = cfg['crop_min'],
                crop_max = cfg['crop_max'],
                only_state=cfg["from_state"],
                num_demos=cfg["num_demos"],
        )
    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*cfg['obs_horizon']
    )

    # example inputs
    noised_action = torch.randn((1, cfg['pred_horizon'], action_dim))
    obs = torch.zeros((1, cfg['obs_horizon'], obs_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    num_epochs = 100

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        model=noise_pred_net,
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=buffer.size * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            # with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            # for nbatch in tepoch:
            batch_points, batch_feats, batch_states, batch_actions, batch_actions_joints = buffer.sample(batch_size)
            # data normalized in dataset
            # device transfer
            nobs = batch_states.to(device)
            naction = batch_actions_joints.to(device)
            B = nobs.shape[0]

            # observation as FiLM conditioning
            # (B, obs_horizon, obs_dim)
            obs_cond = nobs[:,:obs_horizon,:]
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_cond.flatten(start_dim=1)

            # sample noise to add to actions
            noise = torch.randn(naction.shape, device=device)

            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(noise_pred_net)

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

        # Weights of the EMA model
        # is used for inference
        ema_noise_pred_net = ema.averaged_model

        points, feats, state, act = buffer.sample(
            cfg['policy_batch_size'] #, cfg.env.max_episode_steps
        )


    return policy


# @hydra.main(config_path="configs/", config_name="sim_distill", version_base="1.1")
def run_experiment(cfg):
    
    run_path = wandb.init(project="distillation", config=cfg)
    run_path = run_path.path
    os.makedirs(f"checkpoints/{run_path}")
    set_random_seed(cfg['seed'])

    device = "cuda" #f"cuda:{cfg['gpu_id']}"#get_device()
    buffer = Buffer(
                    folder_name=cfg['folder_name'],
                    pad_points=cfg['pad_points'], 
                    voxel_size=cfg['voxel_size'], 
                    num_points=cfg['num_points'], 
                    downsample=cfg['downsample'],
                    crop_min = cfg['crop_min'],
                    crop_max = cfg['crop_max'],
                    only_state=cfg["from_state"],
                    num_demos=cfg["num_demos"],
            )

    env, policy_distill = create_env(cfg)
    # Train policy
    if cfg['from_state']:
        policy_distill = policy_distill.to(device)
    else:
        policy_distill = SparseConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            obs_size=buffer.obs_size,
            act_size=cfg['act_dim'],
            # belief_size=cfg['policy_hidden_size'],
            # hidden_size=cfg['policy_hidden_size'],
            # augment_obs=cfg['augment_obs'],
            # augment_points=cfg['augment_points'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            # emb_layers=cfg['emb_layers'],#[3],
            # emb_size=cfg['emb_size'], #16,
            layers=cfg['layers'],#[32,32]
        ).to(device)

    policy_distill = train_policy(cfg, env, policy_distill, buffer, device, run_path)



MAIN_FOLDER = "/scratch/marcel/data/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default=None)
    parser.add_argument("--usd_name", type=str, default="mugontable.usd")
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=-1)
    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    cfg = config["common"]

    cfg.update(config["teacher_student_distillation"])


    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value

    cfg['folder_name'] = MAIN_FOLDER + "demos/isaac-env/" + args.folder_name 
    if args.num_demos is None:
        cfg['num_demos'] = None
        
    run_experiment(cfg)