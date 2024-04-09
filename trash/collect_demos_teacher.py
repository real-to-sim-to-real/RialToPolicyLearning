import argparse
import wandb
import numpy as np
import yaml
import os
import skvideo.io
from datetime import datetime
from omni.isaac.kit import SimulationApp
import open3d as o3d
import time
from utils import downsample_internal
from rialto.algo.spcnn import SparseConvPolicy
import torch
from utils import create_panda_urdf, add_arm_pcd, downsample_internal, augment_pcds, crop_points_feats, preprocess_points_feats


def run(model_name, run_path,
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        buffer_size=20000, 
        maze_type=0, 
        random_goal=False,
        num_blocks=1, 
        seed=0,
        network_layers="128,128", 
        normalize=False,
        task_config="slide_cabinet,microwave",
        continuous_action_space=False,
        goal_threshold=-1,
        env_name='pointmass_empty',
        num_demos=0,
        max_path_length=100,
        goal_selector_buffer_size=50000,
        gpu=0,
        noise=0,
        img_width=64,
        sensors=["rgb"],
        render_images=False,
        num_cameras=1,
        num_envs=1,
        img_height=64,
        display=False,
        camera_pos="0,0,0", camera_rot="0,0,0,0",
        usd_name="mugandbenchv2",
        demo_name=None,
        camera_pos_rand= [0.1,0.1,0.1],
        camera_target_rand= [0,0.0,0.15],
        camera_target= [0.1,0.1,0.1],
        randomize_pos=False,
        randomize_rot=False,
        randomize_action_mag=None,
        filename=None,
        randomize_object_name="",
        offset=0,
        usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/",
        #todo: ADD ALL OF THE NUM ENVS AND CAMERA PARAMETERS
        goal_selector_name='', **cfg):

    import gym
    import numpy as np
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu
    config = {"headless": not display}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)
    # Envs

    from rialto import envs
    from rialto.envs.env_utils import DiscretizedActionEnv

    # Algo
    from rialto.algo import buffer, variants, networks

    ptu.set_gpu(gpu)

    seed = np.random.choice(100000000)
    print("Seed is", seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, continuous_action_space=continuous_action_space, randomize_object_name=randomize_object_name, randomize_action_mag=randomize_action_mag, display=display, render_images=render_images, img_shape=(img_width,img_height),usd_path=usd_path, usd_name=usd_name, camera_pos=camera_pos, camera_pos_rand=camera_pos_rand, camera_target=camera_target, camera_target_rand=camera_target_rand, num_envs=num_envs, sensors=sensors, num_cameras=num_cameras, randomize_pos=randomize_pos, randomize_rot=randomize_rot)

    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = False
    env_params['fourier'] = fourier
    env_params['fourier_goal_selector'] = fourier_goal_selector
    env_params['normalize']=normalize
    env_params['env_name'] = env_name
    env_params['goal_selector_buffer_size'] = goal_selector_buffer_size
    env_params['input_image_size'] = 64
    env_params['img_width'] = 64
    env_params['img_height'] = 64
    env_params['use_images_in_policy'] = False
    env_params['use_images_in_reward_model'] = False
    env_params['use_images_in_stopping_criteria'] = False
    env_params['close_frames'] = False
    env_params['far_frames'] = False
    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, huge_kwargs = variants.get_params(env, env_params)

    print("run path", run_path)
    print("model_name", model_name)

    huge_kwargs['max_path_length']=max_path_length
    use_goal = False
    if cfg['teacher_images']:
        policy = SparseConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            obs_size=env.observation_space.shape,
            act_size=cfg['act_dim'],
            # belief_size=cfg['policy_hidden_size'],
            # hidden_size=cfg['policy_hidden_size'],
            # augment_obs=cfg['augment_obs'],
            # augment_points=cfg['augment_points'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            # emb_layers=cfg['emb_layers'],#[3],
            # emb_size=cfg['emb_size'], #16,
            layers=cfg['layers'],#[32,32]
            use_state=False
        ).to("cuda")
        print("run path", run_path)
        print("model_name", model_name)
        expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
        policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
        policy = policy.to(f"cuda")
        goal = None

    else:
        if use_goal:
            expert_policy = wandb.restore(f"checkpoint/{model_name}.h5", run_path=run_path)
            policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
            policy = policy.to(f"cuda:{gpu}")
            goal = np.load(wandb.restore(f"checkpoint/commanded_goal_{model_name}.npy", run_path=run_path).name)
            goal = env.extract_goal(goal[0])
        else:
            goal = None
            expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
            policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
            policy = policy.to(f"cuda:{gpu}")
    # goal = np.array(goal)
    now = datetime.now()
    dt_string = usd_name+"_"+now.strftime("%d_%m_%Y_%H:%M:%S")
    if demo_name is not None:
        dt_string = demo_name
    
    if filename is None:
        filename = f"demos/{env_name}/{dt_string}"
    filename = filename + "_" + run_path
    os.makedirs(filename, exist_ok=True)
    print("************Filename", filename)
    cfg["sensors"] = sensors
    collect_demos(env, goal, policy, num_demos, env_name, max_path_length, noise, num_envs, filename, offset, cfg)

# def env_distance(env, state, goal):
#         obs = env.observation(state)
        
#         if isinstance(env.wrapped_env, PointmassGoalEnv):
#             return env.base_env.room.compute_shaped_distance(obs, goal)
#         else:
#             return env.compute_shaped_distance(obs, goal)
def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({video_filename:wandb.Video(images, fps=10)})

def collect_demos(env, goal, policy, num_demos, env_name, max_path_length, noise, num_envs, filename, offset=0,cfg=None, main_folder="/scratch/marcel/data"):
    policy.eval()
    urdf = create_panda_urdf(cfg)

    demo_num = offset
    if cfg['teacher_images']:
        filename = filename + "from_state"
    os.makedirs(f"{main_folder}/{filename}/", exist_ok=True)
    print("filename", f"{main_folder}/{filename}/")
    if demo_num > 0:
        all_actions = list(np.load(f"{main_folder}/{filename}/demo_actions.npy"))[:demo_num]
        all_cont_actions = list(np.load(f"{main_folder}/{filename}/demo_cont_actions.npy"))[:demo_num]
        all_states = list(np.load(f"{main_folder}/{filename}/demo_states.npy"))[:demo_num]
        all_joints = list(np.load(f"{main_folder}/{filename}/demo_joints.npy"))[:demo_num]

    else:
        all_actions =[]
        all_cont_actions = []
        all_states = []
        all_joints = []
        all_pcds_points = []
        all_pcds_images = []
        all_pcds_colors = []
    if goal is not None:
        goal = np.repeat(goal, num_envs).reshape(goal.shape[0], num_envs).T

    while demo_num < num_demos:
        print("Collecting demo number", demo_num)
        actions = []
        states = []
        joints = []
        cont_actions = []

        # goal = env.extract_goal(env.sample_goal())

        state = env.reset()
        joint = env.base_env._env.get_robot_joints()
        images = []
        pcds = []
        start_demo =time.time()
        for t in range(max_path_length):
            start = time.time()
            img, pcd = env.render_image(cfg["sensors"])
            print("Rendering pcd image", time.time()-start)
            observation = env.observation(state)
            if cfg['teacher_images']:            
                # pcd = pcd[0]
                pcd_processed_points = []
                pcd_processed_colors = []
                i = 0
                all_points, all_colors = pcd
                for points in all_points:
                    points_cropped, _ = crop_points_feats(points, np.ones(len(points)), cfg['crop_min'], cfg['crop_max'])
                    points, colors = downsample_internal(points_cropped,  np.ones(len(points)), cfg['num_points'])
                    points = add_arm_pcd(points, joint[i].detach().cpu().numpy(), urdf, cfg['arm_num_points'])
                   
                    pcd_processed_points.append(points)
                    pcd_processed_colors.append(colors)
                    i+= 1

                    # import IPython
                    # IPython.embed()
                    # import open3d as o3d
                    # pcd = o3d.geometry.PointCloud()
                    # pcd.points = o3d.utility.Vector3dVector(points)
                    # o3d.visualization.draw_geometries([pcd])

                pcd_proc_points = np.array(pcd_processed_points)
                pcd_proc_colors = np.array(pcd_processed_colors)

                pcd_proc_points = augment_pcds(pcd_proc_points)
                pcd_processed_points, pcd_processed_colors = preprocess_points_feats(pcd_proc_points, None, cfg['crop_min'],cfg['crop_max'],cfg['voxel_size'], rgb_feats=False,num_points=cfg['num_points'], add_padding=cfg['pad_points'], downsample_points=cfg['downsample']) #self.preprocess_pcd(pcd)
                
                action = policy((pcd_processed_points, pcd_processed_colors, joint)).argmax(dim=1).cpu().numpy()


            else:
                if goal is None:
                    action = policy.act_vectorized(observation, observation)
                else:
                    action = policy.act_vectorized(observation, goal, greedy=False, noise=noise)
                
            images.append(img)    
            actions.append(action)
            states.append(state)
            joints.append(joint.detach().cpu().numpy())
            pcd_points, pcd_colors = pcd

            pcd_points_downsampled = []
            for points in pcd_points:
                points_down, _ = downsample_internal(points, None, 15000)
                pcd_points_downsampled.append(points_down)

            pcd_points_downsampled = np.array(pcd_points_downsampled)

            # pcd_points = downsample_internal(np.array(pcd_points), None, 15000)
            pcd_colors_downsampled = np.ones(pcd_points_downsampled.shape[:-1])
            downsampled_pcd = pcd_points_downsampled, pcd_colors_downsampled
            pcds.append((downsampled_pcd))

            # import IPython
            # IPython.embed()
            # # create plane
            # x  = np.linspace(-1, 1, 100)
            # y  = np.linspace(-1, 1, 100)
            # xv, yv = np.meshgrid(x, y)
            # xv = xv.flatten()
            # yv = yv.flatten()
            # zs = np.zeros_like(xv)
            # pcd_plane_points = np.stack([xv, yv, zs]).T
            # pcd_plane = o3d.geometry.PointCloud()
            # pcd_plane.points = o3d.utility.Vector3dVector(pcd_plane_points)
            # # pcd.append(pcd_plane)

            # pcd_points, pcd_colors = pcd
            # all_pcds = [pcd_plane]
            # for pts in pcd_points:
            #     new_pcd = o3d.geometry.PointCloud()
            #     new_pcd.points = o3d.utility.Vector3dVector(pts)
            #     all_pcds.append(new_pcd)

            # o3d.visualization.draw_geometries(all_pcds)
            state, _, done , info = env.step(action)
            joint = info["robot_joints"]
            cont_action = info["cont_action"].detach().cpu().numpy()
            cont_actions.append(cont_action)
        success = env.base_env._env.get_success().detach().cpu().numpy()
        
        print(f"Trajectory took {time.time() - start_demo}")
        # final_dist_commanded = env_distance(env, states[-1], goal)
        # print("Final distance 1", final_dist_commanded)
        # put actions states into npy file
        

        start = time.time()
        actions = np.array(actions).transpose(1,0)
        cont_actions = np.array(cont_actions).transpose(1,0,2)
        states = env.observation(np.array(states).transpose(1,0,2))
        joints = np.array(joints).transpose(1,0,2)
        if np.sum(np.logical_not(success)) > 0:
            images_bad = np.concatenate(np.array(images).transpose(1,0,2,3,4)[np.logical_not(success)], axis=1)
            create_video(images_bad, f"{env_name}_failed")

        if np.sum(success>0):
            images = np.concatenate(np.array(images).transpose(1,0,2,3,4)[success], axis=1)
            create_video(images, f"{env_name}")

        # for i in range(np.sum(success)):

        if not cfg['teacher_images']:
            for traj in range(num_envs):
                if success[traj] or cfg['save_all']:
                    all_actions.append(actions[traj])
                    all_cont_actions.append(cont_actions[traj])
                    all_states.append(states[traj])
                    all_joints.append(joints[traj])
                    traj_pcd_points = []
                    traj_pcd_feats = []
                    traj_pcd_images = []
                    for i in range(len(pcds)):
                        traj_pcd_points.append(pcds[i][0][traj])
                        traj_pcd_feats.append(pcds[i][1][traj])
                        traj_pcd_images.append(images[i][traj])

                    np.save(f"{main_folder}/{filename}/traj_{demo_num}_points.npy", traj_pcd_points)
                    # np.save(f"{main_folder}/{filename}/traj_{demo_num}/{i}_feats.npy", traj_pcd_feats )
                    np.save(f"{main_folder}/{filename}/traj_{demo_num}_rgb.npy", traj_pcd_images)

                    # all_pcd_points.append(np.array(traj_pcd_points))
                    # all_pcd_feats.append(np.array(traj_pcd_feats))
                    # all_pcd_images.append(np.array(traj_pcd_images))

                    demo_num += 1
        else:
            for traj in range(num_envs):
                if success[traj] or cfg['save_all']:
                    all_actions.append(actions[traj])
                    all_cont_actions.append(cont_actions[traj])
                    all_states.append(states[traj])
                    all_joints.append(joints[traj])

                    demo_num += 1
        # store all pcds as {filename}/demo_pcd_{i}
        np.save(f"{main_folder}/{filename}/demo_actions.npy", all_actions)
        np.save(f"{main_folder}/{filename}/demo_states.npy", all_states)
        np.save(f"{main_folder}/{filename}/demo_joints.npy", all_joints)
        np.save(f"{main_folder}/{filename}/demo_cont_actions.npy", all_cont_actions)
        # np.save(f"{main_folder}/{filename}/all_points.npy",all_pcd_points)
        # np.save(f"{main_folder}/{filename}/all_feats.npy", all_pcd_feats)
        # np.save(f"{main_folder}/{filename}/all_rgb.npy", all_pcd_images)

        # demo_num += np.sum(success)
        wandb.log({"Success": np.mean(success), "Time/Trajectory": time.time() - start_demo, "NumDemos":demo_num})
        print(f"Storing took", time.time() - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--epsilon_greedy_rollout",type=float, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--model_name", type=str, default='best_model_02_04_2023_09:36:41')
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--usd_name",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_cameras", type=int, default=None)
    parser.add_argument("--render_images", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--demo_name", type=str, default=None)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--teacher_images", action="store_true", default=False)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]

    params.update(config[args.env_name])
    params.update(config["teacher_student_distillation"])
    if args.extra_params is not None:
        params.update(config[args.extra_params])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    data_folder_name = f"{args.env_name}_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name + str(args.seed)



    params["data_folder"] = data_folder_name
    wandb.init(project=args.env_name+"demos", name=f"{args.env_name}_demos", config=params, dir="/data/pulkitag/data/marcel/wandb")


    run(**params)
