import time
import argparse
import wandb
import numpy as np
import yaml
import os
from utils import create_env
import os
from os import listdir
from os.path import isfile, join
from rialto.franka.real_franka_env import RealFrankaEnv

def run(
    **cfg):

    if cfg["from_disk"]:
        cfg["render_images"] = False
    env_name = cfg["env_name"]
    demo_folder = cfg["demo_folder"]
    num_demos = cfg["num_demos"]
    max_path_length = cfg["max_path_length"]
    offset = cfg["offset"]
    save_all = cfg["save_all"]

    env, _ = create_env(cfg, cfg['display'], seed=cfg['seed'])
    env = RealFrankaEnv(cam_index=2, hz=1)

    os.makedirs(f"demos/{env_name+demo_folder}", exist_ok=True)
    
    # import IPython
    # IPython.embed()
    # while simulation_app.is_running():
    #     env.base_env._env.sim.step(True)
    # simulation_app.close()

    collect_demos(env, real_env, num_demos, env_name, max_path_length, offset, save_all, demo_folder, cfg=cfg)

def env_distance(env, state, goal):
        obs = env.observation(state)
        
        return env.compute_shaped_distance(obs, goal)
def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({"demos_video_trajectories":wandb.Video(images, fps=10)})

FRANKA_MAP_ACTION = {
    '\x1b[A': 0,# FRONT
    '\x1b[B': 1,# BACK
    '\x1b[D': 2,# LEFT
    '\x1b[C': 3, # RIGHT
    'U': 4,# UP
    'J': 5,# DOWN
    '1': 6,# TILT 1
    '2': 7,# TILT 2
    '3': 8,# TILT 3
    '4': 9,# TILT 4
    '5': 10,# TILT 5
    '6': 11, # TILT 6
    '7': 12,# TILT 7
    '8': 13,# TILT 7
    'O': 14,# OPEN GRIPPER
    'C': 15,# CLOSE GRIPPER 
    '82': 0,# FRONT
    '84': 1,# BACK
    '81': 2,# LEFT
    '83': 3, # RIGHT
    '117': 4,# UP
    '106': 5,# DOWN
    '49': 6,# TILT 1
    '50': 7,# TILT 2
    '51': 8,# TILT 3
    '52': 9,# TILT 4
    '53': 10,# TILT 5
    '54': 11, # TILT 6
    # '55': 12,# TILT 7
    # '56': 13,# TILT 7
    '99': 13,# CLOSE GRIPPER
    '111': 12,# OPEN GRIPPER
    '100': -1,# DISCARD
    '115': -2,# SAVE

}
import cv2

def get_data_foldername(cfg):
    filename = cfg["datafilename"]
    if "datafolder" in cfg:
        datafolder = cfg["datafolder"]
    else:
        datafolder = "/data/pulkitag/misc/marcel/data/"
    
    folder_name = f"{datafolder}/{filename}" #os.path.join(cfg["main_folder"],cfg["filename"])

    return folder_name
def get_num_demos(cfg):

    folder_name = get_data_foldername(cfg)

    onlyfiles = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
    num_demos_disk = len(onlyfiles)

    return num_demos_disk//3

def collect_demos(env, real_env, num_demos, env_name, max_path_length, offset=0, save_all=False, demo_folder="", cfg=None):
    i = offset
    print(env_name+demo_folder)
    state = env.reset()
    real_state = real_env.reset()
    state = env.step(np.array([0]))
    joints = []
    # actions_real = np.load("realworlddemos/demo_0_actions.npy")
    # joints_real = np.load("realworlddemos/demo_0_joints.npy")
    image = env.render_image(sensors=["rgb"])[0][0]

    if cfg["from_disk"]:
        num_demos_disk = get_num_demos(cfg)
    

    global franka_ck_action  # ck -- carb keyboard
    franka_ck_action = None
    if cfg["carb_keyboard"]:
        print(f'Using carb keyboard interface for actions')
        import omni.appwindow
        import carb.input

        FRANKA_MAP_ACTION_CARB = {
            carb.input.KeyboardInput.W : 0,# FRONT
            carb.input.KeyboardInput.S : 1,# BACK
            carb.input.KeyboardInput.A : 2,# LEFT
            carb.input.KeyboardInput.D : 3,# RIGHT
            carb.input.KeyboardInput.Q : 4,# UP
            carb.input.KeyboardInput.E : 5,# DOWN
            carb.input.KeyboardInput.J : 6,# TILT X
            carb.input.KeyboardInput.L : 7,# TILT X
            carb.input.KeyboardInput.I : 8,# TILT Y
            carb.input.KeyboardInput.K : 9,# TILT Y
            carb.input.KeyboardInput.U : 10,# TILT Z
            carb.input.KeyboardInput.O : 11,# TILT Z
            carb.input.KeyboardInput.V : 12,# OPEN
            carb.input.KeyboardInput.C : 13,# CLOSE
            carb.input.KeyboardInput.KEY_1 : -1,# DISCARD
            carb.input.KeyboardInput.KEY_2 : -2,# SAVE
        }

        def on_kb_input(e):
            global franka_ck_action
            if e.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if e.input in FRANKA_MAP_ACTION_CARB.keys():
                    franka_ck_action = FRANKA_MAP_ACTION_CARB[e.input]
                else:
                    franka_ck_action = None
                print(f'\n\nGot {e.input}, output franka action: {franka_ck_action}\n\n')

        app_window = omni.appwindow.get_default_app_window()
        kb = app_window.get_keyboard()
        carb_input = carb.input.acquire_input_interface()
        _ = carb_input.subscribe_to_keyboard_events(kb, on_kb_input)
    
    def get_action_simple(*, video, **kwargs):
        image = env.render_image(sensors=["rgb"])[0][0]
        video.append(image)
        image = image[:,:,::-1]
        cv2.imshow("Window", image)
        key = cv2.waitKey(10250)
        cv2.destroyAllWindows()
        key = str(key)
        print(key)
        action_raw = key

        while action_raw not in  FRANKA_MAP_ACTION:
            action_raw = "3"
            # action_raw = input("Action:")
        print("action_raw", action_raw)
        
        action = FRANKA_MAP_ACTION[action_raw]
        return action

    def get_action_carb_keyboard(noop_flag=None, *args, **kwargs):
        global franka_ck_action
        if franka_ck_action is None:
            # print(f'[Debug] no-op step')
            return noop_flag
        else:
            return franka_ck_action
    
    get_action = get_action_carb_keyboard if cfg["carb_keyboard"] else get_action_simple

    while i < num_demos + offset:
        actions = []
        states = []
        
        state = env.reset()
        real_state = real_env.reset()
        video = []
        goal = env.extract_goal(env.sample_goal())
        print(env.action_space)
        print("Max path length", max_path_length)
        if cfg["from_disk"]:
            # read the number of demos in disk
            # rollout actions
            # collect demos
            # distill policy
            # train PPO

            folder_name = get_data_foldername(cfg)

            num_demos_disk = get_num_demos(cfg)
            traj_idx = np.random.choice(num_demos_disk, 1)[0]
            actions_disk = np.load(f"{folder_name}/actions_0_{traj_idx}.npy")[0]
        t = 0
        while True:
            if t > max_path_length:
                break

            if cfg["from_disk"]:
                if len(actions_disk) <= t:
                    action = -2
                else:
                    action = actions_disk[t]
            else:
                action = get_action(video=video)

            if action is None:  # no-op flag, just allow the viewport to run forward
                env.base_env._env.sim.render()
                continue
            # action = actions_real[t]
            # real_joints = joints_real[t]
            # sim_joints = env.base_env._env.get_robot_joints().detach().cpu().numpy()[0]

            # print("Real robot joint", real_joints)
            # print("Simulation robot joint", sim_joints)
            if action < 0:
                break
            actions.append(action)
            states.append(state[0])

            state, _, done , info = env.step(np.array([action]))
            real_state, _, real_done , real_info = real_env.step(np.array([action]))

            franka_ck_action = None  # ensure we reset the keyboard action
            time.sleep(0.005)
            t += 1  # only increment t when we actually take an action
            print(t)
            # print("Achieved pos", env.base_env._env._robot.get_ee_pose())
            # print("Achieved pos", env.base_env._env.robot.data.root_state_w)
            # if done :
            #     break

        
        if action == -2:
            final_dist_commanded = env_distance(env, states[-1], goal)
            success = env.base_env._env.get_success().detach().cpu().numpy()
            print("Succeeded", success)
            if not cfg["from_disk"] and len(video):  # todo(as) - add a video recording for the carb version as well
                create_video(video, f"{env_name}_{final_dist_commanded}")
            print("Final distance 1", final_dist_commanded)
    
            # put actions states into npy file
            
            actions = np.array(actions)
            states = np.array(states)
            env.plot_trajectories([states], [goal])
            print("Savingn in", env_name+demo_folder)
            np.save(f"demos/{env_name+demo_folder}/demo_{i}_actions.npy", actions)
            np.save(f"demos/{env_name+demo_folder}/demo_{i}_states.npy", states)
            np.save(f"demos/{env_name+demo_folder}/demo_{i}_joints.npy", joints)
            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--epsilon_greedy_rollout",type=float, default=None)
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--usd_name",type=str, default=None)
    parser.add_argument("--usd_path",type=str, default=None)
    parser.add_argument("--img_width", type=int, default=None)
    parser.add_argument("--img_height", type=int, default=None)
    parser.add_argument("--demo_folder", type=str, default="")
    parser.add_argument("--not_randomize", action="store_true", default=False)
    parser.add_argument("--from_disk", action="store_true", default=False)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--datafolder", type=str, default=None)
    parser.add_argument("--datafilename", type=str, default=None)
    parser.add_argument("--distractors",type=str, default="no_distractors")
    parser.add_argument("--carb_keyboard", action="store_true", default=False)


    args = parser.parse_args()

    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]


    params.update(config[args.env_name])


    params.update({'randomize_pos':not args.not_randomize, 'randomize_rot':not args.not_randomize})
    if args.extra_params is not None:
        all_extra_params = args.extra_params.split(",")
        for extra_param in all_extra_params:
            params.update(config[extra_param])

    data_folder_name = f"{args.env_name}_"

    data_folder_name = data_folder_name+"teleop"

    data_folder_name = data_folder_name + str(args.seed)
        
    params.update(config["teleop_params"])
    # force render_images for teleop_params to be False if we're using the carb keyboard interface
    if args.carb_keyboard:
        params["render_images"] = False
    params.update(config["teleop"])
    params.update(config[args.distractors])
    
    
    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value


    params["data_folder"] = data_folder_name
    del params["camera_rot"]

    if "WANDB_DIR" in os.environ:
        # use environment variable if possible
        wandb_dir = os.environ["WANDB_DIR"]
    else:
        # otherwise use argparse
        wandb_dir = params["wandb_dir"]
    wandb.init(project=args.env_name+"teleop_demos", name=f"{args.env_name}_demos", config=params, dir=wandb_dir)


    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
