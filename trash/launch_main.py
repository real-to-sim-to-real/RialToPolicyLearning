from rialto.algo import huge
from numpy import VisibleDeprecationWarning
import doodad as dd
import rialto.doodad_utils as dd_utils
import argparse
import wandb
import yaml
from omni.isaac.kit import SimulationApp

def run(start_frontier = -1,
        frontier_expansion_rate=10,
        frontier_expansion_freq=-1,
        select_goal_from_last_k_trajectories=-1,
        throw_trajectories_not_reaching_goal=False,
        repeat_previous_action_prob=0.8,
        reward_layers="600,600", 
        fourier=False,
        fourier_goal_selector=False,
        command_goal_if_too_close=False,
        display_trajectories_freq=20,
        label_from_last_k_steps=-1,
        label_from_last_k_trajectories=-1,
        contrastive=False,
        k_goal=1, use_horizon=False, 
        sample_new_goal_freq=1, 
        weighted_sl=False, 
        buffer_size=20000, 
        stopped_thresh=0.05, 
        eval_episodes=200, 
        maze_type=0, 
        random_goal=False,
        explore_length=20, 
        desired_goal_sampling_freq=0.0,
        num_blocks=1, 
        deterministic_rollout=False,
        network_layers="128,128", 
        epsilon_greedy_rollout=0, 
        epsilon_greedy_exploration=0.2, 
        remove_last_k_steps=8, 
        select_last_k_steps=8, 
        eval_freq=5e3, 
        expl_noise_std = 1,
        goal_selector_epochs=400,
        stop_training_goal_selector_after=-1,
        normalize=False,
        task_config="slide_cabinet,microwave",
        human_input=False,
        save_videos = True, 
        continuous_action_space=False,
        goal_selector_batch_size=64,
        goal_threshold=-1,
        check_if_stopped=False,
        human_data_file='',
        env_name='pointmass_empty',train_goal_selector_freq=10, 
        distance_noise_std=0,  exploration_when_stopped=True, 
        remove_last_steps_when_stopped=True,  
        goal_selector_num_samples=100, data_folder="data", display_plots=False, render=False,
        explore_episodes=5, gpu=0, sample_softmax=False, seed=0, load_goal_selector=False,
        batch_size=100,  save_buffer=-1, policy_updates_per_step=1,
        select_best_sample_size=1000, max_path_length=50, lr=5e-4, train_with_preferences=True,
        use_oracle=False,
        use_wrong_oracle=False,
        pretrain_goal_selector=False,
        pretrain_policy=False,
        num_demos=0,
        demo_epochs=100000,
        demo_goal_selector_epochs=1000,
        goal_selector_buffer_size=50000,
        fill_buffer_first_episodes=0,
        run_path='',
        policy_name='',
        img_width=64,
        sensors=["rgb"],
        render_images=False,
        num_envs=1,
        img_height=64,
        display=False,
        camera_pos="0,0,0",
        camera_pos_rand= [0.1,0.1,0.1],
        camera_target_rand= [0,0.0,0.15],
        camera_target= [0.1,0.1,0.1],
        usd_name="mugandbenchv2",
        usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/",
        use_images_in_policy=False, use_images_in_reward_model=False, use_images_in_stopping_criteria=False, close_frames=2, far_frames=10,
        max_timesteps=2e-4, goal_selector_name='', euler_rot=False, **extra_params):

    import gym
    import numpy as np
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, gpu)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu


    config = {"headless": not display}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)


    # import omni.isaac.contrib_envs  # noqa: F401
    # import omni.isaac.orbit_envs  # noqa: F401
    # Envs

    from rialto import envs

    # Algo
    from rialto.algo import buffer, variants, networks

    ptu.set_gpu(gpu)
    device = f"cuda:{gpu}"


    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, continuous_action_space=continuous_action_space, display=display, render_images=render_images, img_shape=(img_width,img_height),usd_path=usd_path, usd_name=usd_name, camera_pos=camera_pos, camera_pos_rand=camera_pos_rand, camera_target=camera_target, camera_target_rand=camera_target_rand, num_envs=num_envs, sensors=sensors, euler_rot=euler_rot)

    env_params = envs.get_env_params(env_name)
    env_params['max_trajectory_length']=max_path_length
    env_params['network_layers']=network_layers
    env_params['reward_layers'] = reward_layers
    env_params['buffer_size'] = buffer_size
    env_params['use_horizon'] = use_horizon
    env_params['fourier'] = fourier
    env_params['fourier_goal_selector'] = fourier_goal_selector
    env_params['normalize']=normalize
    env_params['env_name'] = env_name
    env_params['goal_selector_buffer_size'] = goal_selector_buffer_size
    env_params['input_image_size'] = img_width
    env_params['img_width'] = img_width
    env_params['img_height'] = img_height
    env_params['use_images_in_policy'] = use_images_in_policy
    env_params['use_images_in_reward_model'] = use_images_in_reward_model
    env_params['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
    env_params['close_frames'] = close_frames
    env_params['far_frames'] = far_frames
    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, huge_kwargs = variants.get_params(env, env_params)

    if run_path != "":
        expert_policy = wandb.restore(f"checkpoint/{policy_name}.h5", run_path=run_path)
        policy.load_state_dict(torch.load(expert_policy.name, map_location=f"cuda:{gpu}"))
        expert_goal_selector = wandb.restore(f"checkpoint/{goal_selector_name}.h5", run_path=run_path)
        goal_selector.load_state_dict(torch.load(expert_goal_selector.name, map_location=f"cuda:{gpu}"))

    huge_kwargs['lr']=lr
    huge_kwargs['max_timesteps']=max_timesteps
    huge_kwargs['batch_size']=batch_size
    huge_kwargs['max_path_length']=max_path_length
    huge_kwargs['policy_updates_per_step']=policy_updates_per_step
    huge_kwargs['explore_episodes']=explore_episodes
    huge_kwargs['eval_episodes']=eval_episodes
    huge_kwargs['eval_freq']=eval_freq
    huge_kwargs['remove_last_k_steps']=remove_last_k_steps
    huge_kwargs['select_last_k_steps']=select_last_k_steps
    huge_kwargs['continuous_action_space']=continuous_action_space
    huge_kwargs['expl_noise_std'] = expl_noise_std
    huge_kwargs['check_if_stopped'] = check_if_stopped
    huge_kwargs['num_demos'] = num_demos
    huge_kwargs['demo_epochs'] = demo_epochs
    huge_kwargs['demo_goal_selector_epochs'] = demo_goal_selector_epochs
    huge_kwargs['input_image_size'] = 64
    huge_kwargs['use_images_in_policy'] = use_images_in_policy
    huge_kwargs['use_images_in_reward_model'] = use_images_in_reward_model
    huge_kwargs['classifier_model'] = classifier_model
    huge_kwargs['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
    huge_kwargs['device'] = device
    print(huge_kwargs)

    algo = huge.HUGE(
        env,
        policy,
        goal_selector,
        replay_buffer,
        goal_selector_buffer,
        train_with_preferences=train_with_preferences,
        use_oracle=use_oracle,
        save_buffer=save_buffer,
        load_goal_selector=load_goal_selector,
        sample_softmax = sample_softmax,
        display_plots=display_plots,
        fill_buffer_first_episodes=fill_buffer_first_episodes,
        render=render,
        num_envs=num_envs,
        data_folder=data_folder,
        goal_selector_num_samples=goal_selector_num_samples,
        train_goal_selector_freq=train_goal_selector_freq,
        remove_last_steps_when_stopped=remove_last_steps_when_stopped,
        exploration_when_stopped=exploration_when_stopped,
        distance_noise_std=distance_noise_std,
        save_videos=save_videos,
        human_input=human_input,
        epsilon_greedy_exploration=epsilon_greedy_exploration,
        epsilon_greedy_rollout=epsilon_greedy_rollout,
        explore_length=explore_length,
        stopped_thresh=stopped_thresh,
        render_images=render_images,
        weighted_sl=weighted_sl,
        sample_new_goal_freq=sample_new_goal_freq,
        k_goal=k_goal,
        frontier_expansion_freq=frontier_expansion_freq,
        frontier_expansion_rate=frontier_expansion_rate,
        start_frontier=start_frontier,
        select_goal_from_last_k_trajectories=select_goal_from_last_k_trajectories,
        throw_trajectories_not_reaching_goal=throw_trajectories_not_reaching_goal,
        command_goal_if_too_close=command_goal_if_too_close,
        display_trajectories_freq=display_trajectories_freq,
        label_from_last_k_steps=label_from_last_k_steps,
        label_from_last_k_trajectories=label_from_last_k_trajectories,
        contrastive=contrastive,
        deterministic_rollout=deterministic_rollout,
        repeat_previous_action_prob=repeat_previous_action_prob,
        desired_goal_sampling_freq=desired_goal_sampling_freq,
        goal_selector_batch_size=goal_selector_batch_size,
        goal_selector_epochs=goal_selector_epochs,
        use_wrong_oracle=use_wrong_oracle,
        human_data_file=human_data_file,
        stop_training_goal_selector_after=stop_training_goal_selector_after,
        select_best_sample_size=select_best_sample_size,
        pretrain_goal_selector=pretrain_goal_selector,
        pretrain_policy=pretrain_policy,
        env_name=env_name,
        **huge_kwargs
    )
    algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int, default=1)
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--comment", type=str, default='')
    parser.add_argument("--method", type=str, default='huge')
    parser.add_argument("--epsilon_greedy_rollout",type=float, default=None)
    parser.add_argument("--explore_episodes",type=int, default=None)
    parser.add_argument("--max_path_length",type=int, default=None)
    parser.add_argument("--repeat_previous_action_prob",type=float, default=None)
    parser.add_argument("--network_layers", type=str, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--train_goal_selector_freq",type=int, default=None)
    parser.add_argument("--goal_selector_num_samples",type=int, default=None)
    parser.add_argument("--num_demos", type=int, default=None)
    parser.add_argument("--demo_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stop_training_goal_selector_after", type=int, default=None)
    parser.add_argument("--goal_selector_buffer_size", type=int, default=None)
    parser.add_argument("--demo_goal_selector_epochs", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument("--start_frontier", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--desired_goal_sampling_freq", type=float, default=None)
    parser.add_argument("--human_data_file", type=str, default=None)
    parser.add_argument("--run_path", type=str, default=None)
    parser.add_argument("--policy_name", type=str, default=None)
    parser.add_argument("--fill_buffer_first_episodes", type=int, default=None)
    parser.add_argument("--goal_selector_name", type=str, default=None)
    parser.add_argument("--use_images", action="store_true", default=False)
    parser.add_argument("--pretrain_policy", action="store_true", default=False)
    parser.add_argument("--pretrain_goal_selector", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--distance_noise_std",type=float, default=None)
    parser.add_argument("--usd_name",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--render_images", action="store_true", default=False)

    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]


    params.update(config[args.method])
    
    params.update(config[args.env_name])
    params.update(config["teacher_student_distillation"])

    if args.use_images:
        params.update(config["use_images"])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value

    data_folder_name = f"{args.env_name}_"

    wandb_suffix = args.method
    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name + str(args.seed)

    params["data_folder"] = data_folder_name

    comment = args.comment

    if args.use_images:
        comment = "images_"+comment
    wandb.init(project=args.env_name+"_huge", name=f"{args.env_name}_{args.method}_{args.seed}_{args.comment}", config=params)

    print("params before run", params)

    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
