import argparse
import wandb 
import gym
import numpy as np

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu
# import matplotlib.pyplot as plt
# import seaborn as sns
from PIL import Image
from omni.isaac.kit import SimulationApp
# Envs
import yaml
# import wandb
from stable_baselines3.common.vec_env import DummyVecEnv
from rialto.algo.buffer_distillation import OnlineBufferPPO
# from stable_baselines3 import PPO
from rialto.algo.sb3_ppo import PPO
from gym.spaces import Box, Dict
from math import inf
import torch.nn as nn
import os
from os import listdir
from os.path import isfile, join
from utils import unpad_and_downsample

class WrappedIsaacEnv(DummyVecEnv):
    def __init__(self, env, max_path_length=50, from_vision=False, cfg=None):
        self.env = env
        self.from_vision = from_vision
        self.max_path_length = max_path_length
        self.timestep = 0
        self.observation_space = Box(-inf, inf, (self.env.observation_space.shape[0]*2,))
        self.action_space = self.env.action_space
        self.num_envs = self.env.base_env.num_envs

        if self.from_vision:
            self.num_points = {"num_points":cfg["num_points"]}
            self.observation_space = Box(-inf, inf, (self.num_points["num_points"],3))
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew = rew.cpu().numpy()
        state = self.extract_obs(obs)

        if self.from_vision:
            state = self.env.render_image()[1][0]
            state = unpad_and_downsample(state, self.num_points)

        self.timestep += 1

        # self.env.compute_success() # should be returned by en
        done = torch.tensor([self.timestep >= self.max_path_length for i in range(self.num_envs)]).int().cpu().numpy()
        new_info = []
        for success in info['success_rate']:
            new_info.append({
                "episode": {},
                "success_rate": success,
            })
        
        if np.any(done):
            # import IPython
            # IPython.embed()
            print("Episode done")
            wandb.log({"reward": np.mean(rew), "success":torch.mean(info["success_rate"].float())})
            state = self.reset()

        else:
            print(self.timestep, "runnning")

        # state = torch.tensor(state).to("cuda")
        # rew = torch.tensor(rew).to("cuda")
        # done = torch.tensor(done).to("cuda")
        return state, rew, done, new_info

    def extract_obs(self, obs):
        obs = self.env.observation(obs)
        new_obs = np.hstack([obs, obs])
        return new_obs

    def reset(self,):
        self.timestep = 0

        obs = self.env.reset()
        obs = self.extract_obs(obs)
        if self.from_vision:
            obs = self.env.render_image()[1][0]
            obs = unpad_and_downsample(obs, self.num_points)

        return obs


def get_all_demos(cfg, env):
    num_traj = 0
    buffer = OnlineBufferPPO(env.observation_space.shape[0]//2, cfg=cfg)
    filename = cfg["filename"]
    if "datafolder" in cfg:
        datafolder = cfg["datafolder"]
    else:
        datafolder = "/data/pulkitag/misc/marcel/data/"
    
    folder_name = f"{datafolder}/{filename}" #os.path.join(cfg["main_folder"],cfg["filename"])

    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    all_trajs_files = []
    all_trajs_files_demo_num = []
    for file in onlyfiles:
        if "actions" in file:
            # num = int(file.split("_")[-1].split(".")[0])
            simdemo = "demo" in file
            if simdemo:
                postfix = file.split("_")[1]
                # num = int(file.split("_")[-1].split(".")[0])
                all_trajs_files.append(postfix)
                all_trajs_files_demo_num.append(int(postfix.split("_")[0]))
            else:
                postfix = file[8:]
                # num = int(file.split("_")[-1].split(".")[0])
                all_trajs_files.append(postfix)
                all_trajs_files_demo_num.append(int(postfix.split("_")[1].split(".")[0]))

    sorted_idxs = np.argsort(all_trajs_files_demo_num)
    all_trajs_files = np.array(all_trajs_files)[sorted_idxs]

    for idx in range(min(cfg["num_demos"], len(all_trajs_files))):
        traj_postfix = all_trajs_files[idx]

        if simdemo:
            actions = np.load(folder_name+f"/demo_{traj_postfix}_actions.npy")[None]

            cont_actions = np.ones_like(actions)

            states = env.env.observation(np.load(folder_name+f"/demo_{traj_postfix}_states.npy"))[None]

            joints = states.copy()
            all_pcd_points = None
            all_pcd_colors = None 
            
        else:
            actions = np.load(folder_name+f"/actions_{traj_postfix}")
            joints = np.load(folder_name+f"/joints_{traj_postfix}")
            cont_actions = np.ones_like(actions)

            if os.path.exists(folder_name+f"/states_{traj_postfix}"):
                states = np.load(folder_name+f"/states_{traj_postfix}")
            else:
                print("WARNING: No file states was found")
                states = joints.copy()
            if cfg['from_vision'] and os.path.exists(folder_name+f"/pcd_points_{traj_postfix}"):
                all_pcd_points = np.load(folder_name+f"/pcd_points_{traj_postfix}")
                all_pcd_colors = None
            else:
                all_pcd_points = None
                all_pcd_colors = None 
        print("trajidx", traj_postfix)
        

        buffer.add_trajectories(actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors)
    
    return buffer
# class PolicyWrapper(torch.nn.):
#     def __init__(self, policy):
#         self.policy
def run(wandb_run=None,
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        buffer_size=20000, 
        maze_type=0, 
        random_goal=False,
        num_blocks=1, 
        seed=0,
        n_steps=2048,
        network_layers="128,128", 
        normalize=False,
        task_config="slide_cabinet,microwave",
        continuous_action_space=False,
        goal_threshold=-1,
        env_name='pointmass_empty',
        goal_selector_buffer_size=50000,
        gpu=0,
        noise=0,
        save_all=False,
        display=True,
        save_videos=True,
        img_width=128,
        img_height=128,
        usd_path=None,
        usd_name=None,
        num_envs=7,
        camera_pos="0,0,0",
        camera_pos_rand= [0.1,0.1,0.1],
        camera_target_rand= [0,0.0,0.15],
        camera_target= [0.1,0.1,0.1],
        randomize_pos=False,
        randomize_rot=False,
        num_cameras= 1,
        demo_folder="",
        euler_rot=False,
        offset=0,
        max_timesteps=0,
        model_name = None,
        randomize_action_mag=None,
        run_path = None,
        randomize_object_name="",
        bc_loss=False,
        bc_coef=0.0,
        goal_selector_name='', **cfg):

    config = {"headless": not display}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)
    from rialto import envs
    from rialto.envs.utils.parse_cfg import parse_env_cfg

    ptu.set_gpu(gpu)
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_params = envs.get_env_params(env_name)
    env_params['max_path_length']=cfg["max_path_length"]
    env_params['network_layers']=network_layers
    env_params['reward_model_name'] = ''
    env_params['continuous_action_space'] = continuous_action_space
    env_params['use_horizon'] = False
    env_params['fourier'] = False
    env_params['fourier_goal_selector'] = False
    env_params['pick_or_place'] = True
    env_params['normalize']=False
    env_params['env_name'] = env_name
    env_params['goal_selector_name']=""
    env_params['buffer_size']=1000
    env_params['goal_selector_buffer_size'] = 10
    env_params['reward_layers'] = network_layers

    print(env_params)

    if cfg["from_vision"]:
        cfg["sensors"] = ["synthetic_pcd"]
    sub_env = envs.create_env(env_name, max_path_length=cfg["max_path_length"], randomize_action_mag=randomize_action_mag, randomize_object_name=randomize_object_name,num_envs=num_envs,continuous_action_space=continuous_action_space, display=display, render_images=False, img_shape=(img_width,img_height),usd_path=usd_path, usd_name=usd_name, num_cameras=num_cameras,randomize_pos=randomize_pos, randomize_rot=randomize_rot, euler_rot=euler_rot, cfg=cfg)
    env = WrappedIsaacEnv(sub_env, max_path_length=cfg["max_path_length"], from_vision=cfg["from_vision"], cfg=cfg)

    policy_kwargs = dict()
    policy_kwargs['net_arch'] = [int(l) for l in cfg["expert_network"].split(",")]
    policy_kwargs['activation_fn'] = nn.ReLU



    if cfg["from_vision"]:
        policy_type = "Sparse3DPolicy"
    else:
        policy_type = "MlpPolicy"

    batch_size = cfg["ppo_batch_size"]

    if bc_loss:
        bc_buffer = get_all_demos(cfg, env)
        model = PPO(policy_type, env, verbose=2, n_epochs=cfg["ppo_epochs"], ent_coef = 1e-2, n_steps=n_steps, batch_size=batch_size, tensorboard_log=f'runs/{wandb_run.id}', policy_kwargs=policy_kwargs, device="cuda", bc_buffer=bc_buffer, bc_coef=bc_coef, bc_batch_size=cfg["bc_batch_size"], from_vision=cfg["from_vision"])
    else:
        model = PPO(policy_type, env, verbose=2, n_epochs=cfg["ppo_epochs"], ent_coef = 1e-2, n_steps=n_steps, batch_size=batch_size, tensorboard_log=f'runs/{wandb_run.id}', policy_kwargs=policy_kwargs, device="cuda", from_vision=cfg["from_vision"])

    weights = None

    if run_path is not None:
        # TODO: load from ppo
        if cfg['from_ppo']:
            new_model = PPO(policy_type, env, verbose=2, ent_coef = 1e-2, n_steps=n_steps, batch_size=batch_size, tensorboard_log=f'runs/{wandb_run.id}', policy_kwargs=policy_kwargs, device="cuda", from_vision=cfg["from_vision"])
            model_old = wandb.restore(f"checkpoints/{model_name}.zip", run_path=run_path)
            new_model = new_model.load(model_old.name)
            model.policy = new_model.policy
        else:
            expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
            weights = torch.load(expert_policy.name, map_location=f"cuda:0")
            keys = list(weights.keys())
            weights_extractor = {"0.weight":weights[keys[0]], "0.bias":weights[keys[1]], "2.weight":weights[keys[2]], "2.bias":weights[keys[3]]}
            model.policy.mlp_extractor.shared_net.load_state_dict(weights_extractor)
            weights_action = {"weight":weights[keys[4]], "bias":weights[keys[5]]}
            model.policy.action_net.load_state_dict(weights_action)

    # if cfg['start_frozen']:
    #     model.policy.action_net.requires_grad = False
    #     model.policy.mlp_extractor.shared_net.requires_grad = False
    #     for param in model.policy.action_net.parameters():
    #         param.requires_grad = False
    #     for param in model.policy.mlp_extractor.shared_net.parameters():
    #         param.requires_grad = False           


    # policy = policy.to(f"cuda")
    lr = cfg['lr']
    # if cfg['student_from_state']:
    #     lr = 0.0005

    policy_optimizer = torch.optim.AdamW(model.policy.parameters(),
                                     lr=lr,
                                     amsgrad=True)
    policy_optimizer.zero_grad(set_to_none=True)
         
    for idx in range(cfg["pretrain_steps"]):
        policy_loss = 0

        train_loss, val_loss = model.compute_bc_loss()
        train_loss.backward()
        policy_optimizer.step()
        train_loss = train_loss.item()
        policy_optimizer.zero_grad(set_to_none=True)
        wandb.log({"pretrainloss":train_loss})
        print("pretrain_loss", train_loss)
    print("Storing model")
        # Store the actual policy using wandb functions
    step="00"
    model.save(
        os.path.join(
        "checkpoints",f"model_policy_{step}.zip"
        )
    )

    wandb.save(os.path.join(
            "checkpoints",f"model_policy_{step}.zip"
    ))

    # Store policy in our format so it's easy to use after
    mapping = {
        "mlp_extractor.shared_net.0.weight": "net.net.network.0.weight",
        "mlp_extractor.shared_net.0.bias":"net.net.network.0.bias",
        "mlp_extractor.shared_net.2.weight": "net.net.network.2.weight",
        "mlp_extractor.shared_net.2.bias": "net.net.network.2.bias",
        "action_net.weight": "net.net.network.4.weight",
        "action_net.bias": "net.net.network.4.bias",
    }
    # if weights is None:
    new_weights = {}
    # else:
    #     new_weights = weights.copy()
    policy_weights = model.policy.state_dict()
    for key in mapping:
        new_weights[mapping[key]] = policy_weights[key]

    wandb.log({"saving_step": step})
    torch.save(
        new_weights,
        os.path.join(
            "checkpoints", f"policy_finetune_step_{step}.pt"
        ),
    )
    wandb.save(os.path.join(
            "checkpoints", f"policy_finetune_step_{step}.pt"
        ))


    total_timesteps = 0 
    step = 0
    while True:
        intermediate_timesteps = cfg["max_path_length"]*num_envs*10
        print("num intermediate steps", intermediate_timesteps)
        model.learn(
            total_timesteps=intermediate_timesteps
        )
        total_timesteps+=intermediate_timesteps
        print("Storing model")
        # Store the actual policy using wandb functions
        model.save(
            os.path.join(
            "checkpoints",f"model_policy_{step}.zip"
            )
        )

        wandb.save(os.path.join(
                "checkpoints",f"model_policy_{step}.zip"
        ))

        # Store policy in our format so it's easy to use after
        mapping = {
            "mlp_extractor.shared_net.0.weight": "net.net.network.0.weight",
            "mlp_extractor.shared_net.0.bias":"net.net.network.0.bias",
            "mlp_extractor.shared_net.2.weight": "net.net.network.2.weight",
            "mlp_extractor.shared_net.2.bias": "net.net.network.2.bias",
            "action_net.weight": "net.net.network.4.weight",
            "action_net.bias": "net.net.network.4.bias",
        }
        # if weights is None:
        new_weights = {}
        # else:
        #     new_weights = weights.copy()
        policy_weights = model.policy.state_dict()
        for key in mapping:
            new_weights[mapping[key]] = policy_weights[key]

        wandb.log({"saving_step": step})
        torch.save(
            new_weights,
            os.path.join(
                "checkpoints", f"policy_finetune_step_{step}.pt"
            ),
        )
        wandb.save(os.path.join(
                "checkpoints", f"policy_finetune_step_{step}.pt"
            ))

        step += 1

    obs = env.reset()
    
    wandb_run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, default=None)
    parser.add_argument("--usd_name", type=str, default=None)
    parser.add_argument("--usd_path", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=None, required=True)
    parser.add_argument("--decimation", type=int, default=None)
    parser.add_argument("--num_points_demos", type=int, default=9000)
    parser.add_argument("--policy_batch_size", type=int, default=None)
    parser.add_argument("--from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--preload", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--random_augmentation", action="store_true", default=False)
    parser.add_argument("--eval_freq", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=25)
    parser.add_argument("--max_timesteps", type=int, default=200000000)
    parser.add_argument("--policy_train_steps", type=int, default=2000)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--from_ppo", action="store_true", default=False)
    parser.add_argument("--start_frozen", action="store_true", default=False)
    parser.add_argument("--bc_loss", action="store_true", default=False)
    parser.add_argument("--bc_coef", type=float, default=1)
    parser.add_argument("--bc_batch_size", type=int, default=32)
    parser.add_argument("--ppo_batch_size", type=int, default=None)
    parser.add_argument("--ppo_epochs", type=int, default=None)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--from_vision", action="store_true", default=False)
    parser.add_argument("--dense_reward", action="store_true", default=False)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")
    parser.add_argument("--pretrain_steps", type=int, default=None)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    
    cfg = config["common"]

    cfg.update(config["teacher_student_distillation"])
    cfg.update(config["no_distractors"])
    cfg.update(config[args.pcd_randomness])

    if args.num_demos is None:
        cfg['num_demos'] = None
    
    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            cfg.update(config[extra_param])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            cfg[key] = value

    if args.from_vision:
        cfg.update(config["ppo_vision"])


    if "WANDB_DIR" in os.environ:
        # use environment variable if possible
        wandb_dir = os.environ["WANDB_DIR"]
    else:
        # otherwise use argparse
        wandb_dir = cfg["wandb_dir"]
    wandb_run = wandb.init(project=cfg['usd_name']+"ppo-finetune", name=f"{cfg['usd_name']}_ppo", config=cfg, dir=wandb_dir)
    cfg['wandb_run'] = wandb_run
    run(**cfg)
