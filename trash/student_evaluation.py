from rialto.algo import huge
import argparse
import wandb
import numpy as np
import yaml
from omni.isaac.kit import SimulationApp
import open3d as o3d
import time
from rialto.algo.spcnn import SparseConvPolicy
import torch
from utils import create_panda_urdf, add_arm_pcd, downsample_internal, augment_pcds, crop_points_feats, preprocess_points_feats

urdf = create_panda_urdf()

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
        sensors=["rgb", "pointcloud"],
        render_images=False,
        num_cameras=1,
        num_envs=1,
        img_height=64,
        display=False,
        camera_pos="0,0,0",
        camera_pos_rand= [0.1,0.1,0.1],
        camera_target_rand= [0,0.0,0.15],
        camera_target= [0.1,0.1,0.1],
        usd_name="mugandbenchv2",
        demo_name=None,
        from_state=False,
        randomize_pos=False,
        randomize_rot=False,
        randomize_object_name="",
        randomize_action_mag=None,
        use_state=False,
        usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/",
        #todo: ADD ALL OF THE NUM ENVS AND CAMERA PARAMETERS
        goal_selector_name='', **cfg):

    import gym
    import numpy as np
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)
    device = f"cuda"
    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu
    config = {"headless": not display}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)
    # Envs

    from rialto import envs

    # Algo
    from rialto.algo import variants 

    ptu.set_gpu(gpu)


    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, continuous_action_space=continuous_action_space,randomize_object_name=randomize_object_name, randomize_action_mag=randomize_action_mag, display=display, render_images=render_images, img_shape=(img_width,img_height),usd_path=usd_path, usd_name=usd_name, camera_pos=camera_pos, camera_pos_rand=camera_pos_rand, camera_target=camera_target, camera_target_rand=camera_target_rand, num_envs=num_envs, sensors=sensors, num_cameras=num_cameras, randomize_pos=randomize_pos, randomize_rot=randomize_rot)

    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']="256,256"
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = False
    env_params['fourier'] = False
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
    
    if not from_state:
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
            use_state=use_state
        ).to(device)
    print("run path", run_path)
    print("model_name", model_name)
    expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
    policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
    policy = policy.to(f"cuda")
    huge_kwargs['max_path_length']=max_path_length

    cfg['num_envs'] = num_envs
    evaluate_policy(env, policy, num_demos, env_name, max_path_length, cfg=cfg, from_state=from_state, use_state=use_state)


def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({video_filename:wandb.Video(images, fps=10)})

def evaluate_policy(env, policy, num_demos, env_name, max_path_length, cfg, from_state, use_state):
    policy.eval()
    demo_num = 0
    # all_actions = []
    # all_states = []
    # goal = np.repeat(goal, num_envs).reshape(goal.shape[0], num_envs).T
    state = env.reset()
    env.step(np.zeros(env.base_env.num_envs).astype(np.int))
    img, pcds = env.render_image(["pointcloud", "rgb"])

    while demo_num < num_demos:
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
        print("max path length")
        for t in range(max_path_length):
            start = time.time()
            if from_state:
                img = env.render_image(sensors)
            else:
                img, pcds = env.render_image(sensors)
            
                # pcd = pcd[0]
                pcd_processed_points = []
                pcd_processed_colors = []
                i = 0
                all_points, all_colors = pcds
                for points in all_points:
                    points_cropped, _ = crop_points_feats(points, np.ones(len(points)), cfg['crop_min'], cfg['crop_max'])
                    points, colors = downsample_internal(points_cropped,  np.ones(len(points)), cfg['num_points'])
                    points = add_arm_pcd(points, joints[i].detach().cpu().numpy(), urdf, cfg['arm_num_points'])
                   
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

                
            images.append(img)
            print("Rendering pcd image", time.time()-start)
            observation = env.observation(state)

            if from_state:
                observation = torch.tensor(observation)
                action = policy(observation, observation).argmax(dim=1).cpu().numpy()
            else:
                # import IPython
                # IPython.embed()
                # pcd = o3d.geometry.PointCloud()
                # from huge.algo.spcnn import unpad_points
                # pcd.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points[0]))
                # o3d.visualization.draw_geometries([pcd])
                action = policy((pcd_processed_points, pcd_processed_colors, joints)).argmax(dim=1).cpu().numpy()
            # import IPython
            # IPython.embed()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector((pcd_processed_points[1]))
            # o3d.visualization.draw_geometries([pcd])
            # if not from_state:
            #     np.save(f"pcd_timestep_{t}", pcd_processed_points)

            actions.append(action)
            states.append(state)
            
            #store pointcloud in traj_i
            state, _, done , info = env.step(action)
            joints = info["robot_joints"]

        # store action sequence
        # import IPython
        # IPython.embed()
        np.save("all_actions.npy", actions)
        success = env.base_env._env.get_success().detach().cpu().numpy()
        wandb.log({"Success": np.mean(success)})
        images = np.array(images).transpose(1,0,2,3,4)
        if np.sum(success)>0:
            images_success = np.concatenate(images[success], axis=1)
            create_video(images_success, "success")

        if np.sum(success)< success.shape[0]:
            images_failed = np.concatenate(images[np.logical_not(success)], axis=1)
            create_video(images_failed, "failed")

        print(f"Trajectory took {time.time() - start_demo}")
        demo_num += cfg['num_envs']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--epsilon_greedy_rollout",type=float, default=None)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--model_name", type=str, default='best_model_02_04_2023_09:36:41')
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--usd_name",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--num_cameras", type=int, default=None)
    parser.add_argument("--render_images", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--demo_name", type=str, default=None)
    parser.add_argument("--from_state", action="store_true", default=False)
    parser.add_argument("--use_state", action="store_true", default=False)
    parser.add_argument("--fixed", action="store_true", default=False)
    parser.add_argument("--extra_params", type=str, default=None)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]

    params.update(config[args.env_name])
    params.update(config["teacher_student_distillation"])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    data_folder_name = f"{args.env_name}_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name + str(args.seed)

    params["data_folder"] = data_folder_name

    params.update(config['teacher_student_distillation'])
    params.update({'randomize_pos':not args.fixed})
    params.update({'randomize_rot':not args.fixed})
    params.update({'num_envs': args.num_envs})
    if args.extra_params is not None:
        params.update(config[args.extra_params])
    # params["randomize_action_mag"] = np.array([0.01, 0.01, 0.01, 0.03,0.03, 0.03])
    wandb.init(project=args.env_name+"studen_eval", name=f"{args.env_name}_student_eval", config=params, dir="/data/pulkitag/data/marcel/wandb")

    
    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
