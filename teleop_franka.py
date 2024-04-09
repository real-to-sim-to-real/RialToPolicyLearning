import argparse
import wandb
import numpy as np
import yaml
import os
from rialto.franka.real_franka_env import RealFrankaEnv
from utils import downsample_internal
from rialto.algo.buffer_distillation import OnlineBuffer
from utils import create_panda_urdf, preprocess_pcd, postprocess_pcd, preprocess_pcd_from_canon
def run(

        env_name='pointmass_empty',
        num_demos=0,
        max_path_length=100,
        img_width=128,
        img_height=128,
        demo_folder="",
        offset=0,
        cam_index=2,
        **cfg):

    env = RealFrankaEnv(cam_index=cam_index, hz=cfg['hz'], cfg=cfg)
    
    collect_demos(env, num_demos, max_path_length, offset, demo_folder, cfg=cfg)

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
    'u': 4,# UP
    'j': 5,# DOWN
    '1': 6,# TILT 1
    '2': 7,# TILT 2
    '3': 8,# TILT 3
    '4': 9,# TILT 4
    '5': 10,# TILT 5
    '6': 11, # TILT 6
    '7': -3,# TILT 7
    '8': -3,# TILT 7
    'O': 12,# OPEN GRIPPER
    'o': 12,# OPEN GRIPPER
    'C': 13,# CLOSE GRIPPER 
    'c': 13,# CLOSE GRIPPER 
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
    'D': -1,# DISCARD
    'd': -1,# DISCARD
    'S': -2,# SAVE
    's': -2,# SAVE

}
import cv2
def collect_demos(env, num_demos, max_path_length, offset=0, filename="realworlddemo", main_folder="/home/marcel/data", cfg=None):
    i = offset
    env_name = "real_world_pusher"
    # sim_actions = np.load("all_actions_open_final_drawer.npy")

    demo_num = offset
    os.makedirs(f"{main_folder}/{filename}/", exist_ok=True)

    urdf = create_panda_urdf(cfg)

    sensors = ['rgb', 'pointcloud']

    while demo_num < num_demos:
        actions = []
        states = []
        joints = []
        all_pcd_points = []

        
        state = env.reset()
        import IPython
        IPython.embed()
        joint = env._env.get_robot_joints()
        video = []
        images = []
        pcds = []

        print(env.action_space)
        print("Max path length", max_path_length)
        for t in range(max_path_length):
            print("state", t, state)
            img, pcd = env.render_image(sensors)
            points, colors = pcd
            points = points.reshape((1, -1, 3))
            colors = colors.reshape((1, -1, 3))
            if cfg["presample_arm_pcd"]:
                # this is faster (doesn't re-sample the points from the mesh each time)
                pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
            else:
                pcd_processed_points = preprocess_pcd(pcd, joint, urdf, cfg)
            # pcd_processed_points = preprocess_pcd(pcd, joint, urdf, cfg)
            pcd_processed_points_full, pcd_processed_colors_full = postprocess_pcd(pcd_processed_points, cfg) 
            # pcd_processed_points_full, pcd_processed_colors_full, pcd_processed_points, pcd_processed_colors = process_pcd(pcd, joint, urdf, cfg)

            # import IPython
            # IPython.embed()
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pcd_processed_points_full[0])
            # o3d.visualization.draw_geometries([pcd])

            video.append(img)
            # cv2.imshow("Window", image)
            # key = cv2.waitKey(10250)
            key = input("Action:")
            # cv2.destroyAllWindows()
            key = str(key)
            print(key)
            action_raw = key

            while action_raw not in  FRANKA_MAP_ACTION:
                action_raw = input("Action:")
            print("action_raw", action_raw)
            action = FRANKA_MAP_ACTION[action_raw]
            print("action mapped", action)
            if action < 0:
                    break

            images.append(img)    
            actions.append(action)
            states.append(state.detach().cpu().numpy())
            joints.append(joint[0].detach().numpy())
            pcd_points, pcd_colors = pcd

            # pcd_points_downsampled, _ = downsample_internal(pcd_points, None, 15000)

            # # pcd_points = downsample_internal(np.array(pcd_points), None, 15000)
            # pcd_colors_downsampled = np.ones(pcd_points_downsampled.shape[:-1])
            # downsampled_pcd = pcd_points_downsampled, pcd_colors_downsampled
            all_pcd_points.append(pcd_processed_points)

            # action = sim_actions[t,1]
            state, _, done , info = env.step(np.array([action]))
            joint = env._env.get_robot_joints()

            # if done :
            #     break

        if action == -2:
            buffer = OnlineBuffer(cfg=cfg)
            actions = np.array(actions)[None, ...]
            states = np.array(states)[None, ...]
            joints = np.array(joints)[None, ...]
            all_pcd_points = np.array(all_pcd_points).transpose(1,0,2,3)
            buffer.add_trajectories(actions, np.ones_like(actions), states, joints, all_pcd_points, np.ones_like(all_pcd_points), None)
            buffer.store(demo_num, filename, main_folder="demos")
            create_video(video[0], f"{env_name}_{i}")
    
            demo_num += 1
            # assert False

            # put actions states into npy file

            # traj_pcd_points = []
            # traj_pcd_feats = []
            # traj_pcd_images = []
            # for i in range(len(pcds)):
            #     traj_pcd_points.append(pcds[i][0])
            #     traj_pcd_feats.append(pcds[i][1])
            #     traj_pcd_images.append(images[i])

            # np.save(f"{main_folder}/{filename}/traj_{demo_num}_points.npy", traj_pcd_points)
            # # np.save(f"{main_folder}/{filename}/traj_{demo_num}/{i}_feats.npy", traj_pcd_feats )
            # np.save(f"{main_folder}/{filename}/traj_{demo_num}_rgb.npy", traj_pcd_images)



            # # env.plot_trajectories([states], [goal])
            # print("Saving in", f"{main_folder}/{filename}")
            # np.save(f"{main_folder}/{filename}/demo_{demo_num}_actions.npy", actions)
            # np.save(f"{main_folder}/{filename}/demo_{demo_num}_states.npy", states)
            # np.save(f"{main_folder}/{filename}/demo_{demo_num}_joints.npy", joints)

    env.reset()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--num_demos", type=int, default=10)
    parser.add_argument("--max_path_length", type=int, default=None)
    parser.add_argument("--save_all", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--img_width", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=512)
    parser.add_argument("--demo_folder", type=str, default="realworldfranka")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--hz", type=float, default=1)
    parser.add_argument("--total_loop_time", type=float, default=0)
    # parser.add_argument("--hz", type=float, default=0)
    parser.add_argument("--cam_index", type=int, default=2)
    parser.add_argument("--interp_steps", type=int, default=0)
    parser.add_argument("--start_interp_offset", type=int, default=0)
    parser.add_argument("--extra_params", type=str, default=None)
    parser.add_argument("--pcd_randomness",type=str, default="default_pcd_randomness")

    args = parser.parse_args()
    params = {}
    with open("config.yaml") as file:
        config = yaml.safe_load(file)

    params = config["common"]

    params.update(config["isaac-env"])

    data_folder_name = f"isaac_env_"

    data_folder_name = data_folder_name+"_use_oracle_"

    data_folder_name = data_folder_name

    params["data_folder"] = data_folder_name


    params.update(config['teacher_student_distillation'])
    params.update({"max_path_length":args.max_path_length})
    params.update(config[args.pcd_randomness])

    if args.extra_params is not None:
        extra_params = args.extra_params.split(",")
        for p in extra_params:
            params.update(config[p])

    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value


    params["data_folder"] = data_folder_name

    wandb.init(project="real_franka"+"teleop_demos", name=f"real_franka_demos", config=params, dir="/data/pulkitag/data/marcel/wandb")


    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
