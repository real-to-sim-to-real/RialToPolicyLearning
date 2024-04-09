from re import I
from numpy import VisibleDeprecationWarning
from rialto.algo import huge
import argparse
import wandb
import io
import imageio as iio
from PIL import Image
from fastapi import Response, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.responses import PlainTextResponse
from omni.isaac.kit import SimulationApp

app = FastAPI()
app.add_middleware(CORSMiddleware, 
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

global start
start = 0

global algo
algo = None

def start_and_train_algo():
    global algo
    algo = run(**params_algo)
    algo.train()

class QuestionResponse:
    def __init__(self, question_id) -> None:
        self.question_id = question_id
        pass

@app.get("/image", response_class=Response)
async def get_image(questionId =None, background_tasks:BackgroundTasks=None):
    print("Get image Question Id", questionId)
    if algo is not None:
        im = algo.get_image_for_question(questionId)
    else:
        im = np.zeros((64,64,3))
    with io.BytesIO() as buf:
        #iio.imwrite(buf, im, plugin="pillow", format="PNG")
        Image.fromarray(im).save(buf, format="PNG")
        im_bytes = buf.getvalue()
        
    headers = {'Content-Disposition': 'inline; filename="robot.png"', "CrossOrigin":"Anonymous"}
    return Response(im_bytes, headers=headers, media_type='image/png')

@app.get("/start", response_class=Response)
async def start_algo(background_tasks:BackgroundTasks=None):
    global start
    start += 1

    print("Calling start algo")

    if start == 1:
        print("Starting algo")
        background_tasks.add_task(start_and_train_algo)


@app.get("/answer_question", response_class=PlainTextResponse)
async def answer_question(answer=None, questionId=None):
    print("Answer question", answer)
    if answer == "right":
        label = 1
    elif answer == "left":
        label = 0
    else:
        label=None

    print("label", label)
    new_q_id = -1
    if not algo is None:
        new_q_id = algo.answer_question(label, questionId)
        
    print("The answer is ", new_q_id)
    headers = {"CrossOrigin":"Anonymous"}

    return PlainTextResponse(str(new_q_id), headers=headers)
    
from fastapi import Header
import numpy as np 
import skvideo


async def get_video(answer =None, background_tasks:BackgroundTasks=None):
    print("answer video", answer)

    global start
    start += 1

    video = algo.current_video

    with io.BytesIO() as buf:
        iio.imwrite(buf, video, plugin="pillow", format="PNG")
        im_bytes = buf.getvalue()
            
    headers = {'Content-Disposition': 'inline; filename="robot.png"', "CrossOrigin":"Anonymous"}
    return Response(im_bytes, headers=headers, media_type='image/png')
    
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
        render_images = False,
        num_demos=0,
        demo_epochs=100000,
        demo_goal_selector_epochs=1000,
        goal_selector_buffer_size=50000,
        fill_buffer_first_episodes=0,
        num_envs=2,
        img_width=64,
        img_height=64,
        demo_folder="",
        display=False,
        camera_pos="0,0,0", 
        camera_pos_rand= [0.1,0.1,0.1],
        camera_target_rand= [0,0.0,0.15],
        camera_target= [0.1,0.1,0.1],
        randomize_pos=False,
        randomize_rot=False,
        usd_name="mugandbenchv2",
        usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/",
        use_images_in_policy=False, use_images_in_reward_model=False, use_images_in_stopping_criteria=False, close_frames=2, far_frames=10,
        max_timesteps=2e-4, goal_selector_name='', euler_rot=False, num_demos_goal_selector=-1, **extra_params):
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


    import omni.isaac.contrib_envs  # noqa: F401
    import omni.isaac.orbit_envs  # noqa: F401
    # Envs

    from rialto import envs

    # Algo
    from rialto.algo import buffer, variants, networks

    ptu.set_gpu(gpu)
    device = f"cuda:{gpu}"


    torch.manual_seed(seed)
    np.random.seed(seed)
    env = envs.create_env(env_name, continuous_action_space=continuous_action_space, display=display, render_images=render_images, img_shape=(img_width,img_height),usd_path=usd_path, usd_name=usd_name, camera_pos=camera_pos, camera_pos_rand=camera_pos_rand, camera_target=camera_target,randomize_pos=randomize_pos, randomize_rot=randomize_rot, camera_target_rand=camera_target_rand, euler_rot=euler_rot, num_envs=num_envs)

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
    env_params['img_width'] = img_width
    env_params['img_height'] = img_height
    env_params['input_image_size'] = img_width
    env_params['goal_selector_buffer_size'] = goal_selector_buffer_size
    env_params['use_images_in_policy'] = use_images_in_policy
    env_params['use_images_in_reward_model'] = use_images_in_reward_model
    env_params['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
    env_params['close_frames'] = close_frames
    env_params['far_frames'] = far_frames
    print(env_params)
    env_params['goal_selector_name']=goal_selector_name
    env_params['continuous_action_space'] = continuous_action_space
    env, policy, goal_selector, classifier_model, replay_buffer, goal_selector_buffer, huge_kwargs = variants.get_params(env, env_params)

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
    huge_kwargs['input_image_size'] = img_width
    huge_kwargs['use_images_in_policy'] = use_images_in_policy
    huge_kwargs['use_images_in_reward_model'] = use_images_in_reward_model
    huge_kwargs['classifier_model'] = classifier_model
    huge_kwargs['use_images_in_stopping_criteria'] = use_images_in_stopping_criteria
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
        num_demos_goal_selector=num_demos_goal_selector,
        demo_folder=demo_folder,
        **huge_kwargs
    )

    return algo

env_name = os.getenv("ENV_NAME", "kitchenSeq")
num_demos = int(os.getenv("NUM_DEMOS", "0"))
demo_epochs = int(os.getenv("DEMO_EPOCHS", "1000"))
demo_folder = os.getenv("DEMO_FOLDER", "")
usd_name = os.getenv("USD_NAME", "")
max_path_length = int(os.getenv("MAX_PATH_LENGTH", "35"))
render_images = int(os.getenv("RENDER_IMAGES", "0"))
display = bool(int(os.getenv("VISUALIZE", "0")))
fixed = bool(int(os.getenv("FIXED", "0")))
network_layers = os.getenv("NETWORK_LAYERS", "400,600,600,300")
num_envs = int(os.getenv("NUM_ENVS", "1"))
human_data_file = os.getenv("HUMAN_DATA_FILE", "")
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
print("Env name", env_name)

import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)

params = config["common"]

params.update({'env_name':env_name})
params.update(config["human"])
params.update(config[env_name])
params.update({'num_demos':num_demos})
params.update({'demo_epochs':demo_epochs})
params.update({'demo_folder':demo_folder})
params.update({'usd_name':usd_name})
params.update({'max_path_length':max_path_length})
params.update({'human_data_file':human_data_file})
params.update({'render_images':render_images})
params.update({'display':display})
params.update({'num_envs':num_envs})
params.update({'network_layers':network_layers})
params.update({'randomize_pos':not fixed})
params.update({'randomize_rot':not fixed})
params.update({'select_last_k_steps':1000}) # TODO remove
# params.update(config['finetune_alone'])

print("Max path length", max_path_length)
wandb_suffix = "human"

data_folder_name = env_name + "human"

params["data_folder"] = data_folder_name

wandb.init(project=env_name+"huge_human_interface", name=f"{env_name}", config=params)

print("params before run", params)

global params_algo
params_algo = params
