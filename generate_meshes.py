import argparse
import wandb
import numpy as np
import yaml
import os
from utils import create_env
import os
from os import listdir
from os.path import isfile, join

def generate_all_meshes(cfg, env, visualize=False, generate_arm_mesh=False):
    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.orbit.utils.math import quat_apply

    mesh_prim_paths = []

    for object_name in cfg["mesh_names"]:
        object_num = object_name.split("_")[-1]
        mesh_prim_paths.append(f"Xform_{object_num}/Object_{object_num}")
        print(mesh_prim_paths[-1])

    folder = cfg["foldermeshname"] #"booknshelvemesh"
    os.makedirs(folder, exist_ok=True)
    os.makedirs("franka_arm_meshes", exist_ok=True)

    import IPython
    IPython.embed()
    # while True:
    #     env.step(np.array([0]))
    
    import torch
    import trimesh
    from omni.isaac.core.prims import RigidPrimView

    all_arm_points = []
    visualize = True

    if generate_arm_mesh:
        arm_mesh_prim_path = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 
        'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
        
        views  = []
        for prim_name in arm_mesh_prim_path:

            view = RigidPrimView(
                        prim_paths_expr=f"/World/envs/env_*/Robot/{prim_name}", reset_xform_properties=False
            )
            view.initialize()
            views.append(view)
        
    

        i= 0
        for prim_name in arm_mesh_prim_path:
            print(prim_name)
            mesh_prim_path = f'/World/envs/env_0/Robot/{prim_name}/visuals/{prim_name}'  # leaf level prim
            mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
            points_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'points'))
            faces_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'faceVertexIndices')).reshape(-1, 3)
            scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'xformOp:scale'))
            # if "184" in mesh_prim_path:
            points_scaled_np = points_np * scale_np

            tmesh = trimesh.Trimesh(vertices=points_scaled_np, faces=faces_np)

            points = tmesh.sample(5000)

            np.save(f"franka_arm_meshes/{prim_name}_pcd.npy", points)

        pose = views[i].get_world_poses()
        pos = pose[0]
        rot = pose[1]
        print(pos, rot)

        rotated_points = quat_apply( rot, torch.tensor(points).to("cuda").float())
        trans_points = rotated_points + pos
        all_arm_points.append(trans_points.cpu().detach().numpy())
        i+=1

    for prim_name in mesh_prim_paths:
        mesh_prim_path = f'/World/envs/env_0/Scene/{prim_name}/Geometry/Object_Geometry'  # leaf level prim
        mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
        object_prim_path = f'/World/envs/env_0/Scene/{prim_name}'  # leaf level prim
        object_prim = prim_utils.get_prim_at_path(object_prim_path)
        points_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'points'))
        faces_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'faceVertexIndices')).reshape(-1, 3)
        # root_prim = prim_utils.get_prim_at_path(f'/World/envs/env_0/Scene/Xform_184/Object_184/Geometry/Object_Geometry')
        # scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(root_prim), 'xformOp:scale'))
        transform = np.array(prim_utils.get_prim_property(prim_utils.get_prim_path(object_prim), 'xformOp:transform'))
        transform[-1,:] = [0,0,0,1]
        transform_scale = np.zeros((4,4))
        transform_scale[0,0] = transform[0,0]
        transform_scale[1,1] = transform[1,1]
        transform_scale[2,2] = transform[2,2]
        transform_scale[3,3] = transform[3,3]
        from pxr import UsdGeom
        import torch
        xformable = UsdGeom.Xformable(mesh_prim)
        transform_matrix = np.array(xformable.GetLocalTransformation())
        print("Transform matrix", transform_matrix, transform)
        expand_points = np.concatenate([points_np, np.ones((len(points_np),1))], axis=1)
        points_inv = (expand_points@transform_matrix@transform)[:,:3]

        # rotation = R.from_quat(object_rot)
        # rotated_points = rotation.apply(points)
        # rotation_quat = torch.tensor([ 0, 0.6335811, 0.6335811, 0.4440158]).to("cuda")
        rotation_quat1 = torch.tensor([ 0, 0, 0.701, 0.701]).to("cuda")
        rotation_quat2 = torch.tensor([ 0, 0, 0, 1.0]).to("cuda")
        points_inv = quat_apply( rotation_quat1, torch.tensor(points_inv).to("cuda").float()).cpu()
        points_inv = quat_apply( rotation_quat2, torch.tensor(points_inv).to("cuda").float()).cpu()
        data_name = prim_name.split("/")[-1]

        # print("transform", transform)
        # scale = [transform[0,0],transform[1,1],transform[2,2]]
        # if "184" in mesh_prim_path:
        #     scale = [3,3,3]
        # print("Transform matrix", transform_matrix, transform)
        # expand_points = np.concatenate([points_inv, np.ones((len(points_inv),1))], axis=1)
        # points_scaled_np = (expand_points@np.array(transform))[:,:3]
        # points_scaled_np = points_inv * np.array(scale)
        points_scaled_np = points_inv
        tmesh = trimesh.Trimesh(vertices=points_scaled_np, faces=faces_np)

        points = tmesh.sample(5000)
        points_inv = points



        np.save(f"{folder}/{data_name}_pcd.npy", points_inv)


        data = env.base_env._env.env.scene._data

        object_pos = data[data_name].root_state_w[:,:3].clone()
        # mid = object_pos[:,1].clone()
        # object_pos[:,1] = object_pos[:,2]
        # object_pos[:,2] = -mid
        object_rot = data[data_name].root_state_w[:,3:7]
        print("object pos",data_name, object_pos, object_rot)
        from scipy.spatial.transform import Rotation as R
        # rotation = R.from_quat(object_rot)
        # rotated_points = rotation.apply(points)
        # rotated_points = quat_apply( object_rot, torch.tensor(points_inv).to("cuda").float())
        rotated_points = torch.tensor(points_inv).to("cuda").float()
        trans_points = rotated_points + object_pos
        all_arm_points.append(trans_points.cpu().detach().numpy())
        # all_arm_points.append(points_inv)

        # mesh_prim_path = f'/World/envs/env_0/Scene/{prim_name}/Geometry/Object_Geometry'  # leaf level prim
        # mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
        # points_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'points'))
        # faces_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'faceVertexIndices')).reshape(-1, 3)
        # # root_prim = prim_utils.get_prim_at_path(f'/World/envs/env_0/Scene/Xform_184/Object_184/Geometry/Object_Geometry')
        # # scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(root_prim), 'xformOp:scale'))
        # scale = [1,1,1]
        # import IPython
        # IPython.embed()
        # scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'xformOp:scale'))
        # if scale_np is None:
        #     scale = [1,1,1]
        # if "184" in mesh_prim_path:
        #     scale = [3,3,3]
        # points_scaled_np = points_np * np.array(scale)

        # tmesh = trimesh.Trimesh(vertices=points_scaled_np, faces=faces_np)

        # points = tmesh.sample(5000)
        

        # from pxr import UsdGeom
        # import torch
        # xformable = UsdGeom.Xformable(mesh_prim)
        # transform_matrix = xformable.GetLocalTransformation()

        # expand_points = np.concatenate([points, np.ones((len(points),1))], axis=1)
        # points_inv = (expand_points@np.array(transform_matrix))[:,:3]
        # data_name = prim_name.split("/")[-1]
        
        # np.save(f"{folder}/{data_name}_pcd.npy", points_inv)
        
        
        # data = env.base_env._env.env.scene._data
        
        # object_pos = data[data_name].root_state_w[:,:3]
        # object_rot = data[data_name].root_state_w[:,3:7]
        
        # from scipy.spatial.transform import Rotation as R
        # # rotation = R.from_quat(object_rot)
        # # rotated_points = rotation.apply(points)
        # rotated_points = quat_apply( object_rot, torch.tensor(points_inv).to("cuda").float())
        # trans_points = rotated_points + object_pos
        # all_arm_points.append(trans_points.cpu().detach().numpy())

    
    if visualize:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_arm_points))
        o3d.visualization.draw_geometries([pcd])


     
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
    save = False

    env, _ = create_env(cfg, cfg['display'], seed=cfg['seed'])
    obs = env.reset()
    img = env.render(["pointcloud"])

    import omni.isaac.core.utils.prims as prim_utils
    from omni.isaac.orbit.utils.math import quat_inv, quat_apply, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz

    import IPython
    IPython.embed()


    root_object_prim = ''  # top level prim of the object
    mesh_prim_paths = [ "Xform_268/Object_268", "Xform_272/Object_272", "Xform_277/Object_277"]

    for object_name in cfg["mesh_names"]:
        object_num = object_name.split("_")[-1]
        mesh_prim_paths.append(f"Xform_{object_num}/Object_{object_num}")
        print(mesh_prim_paths[-1])

    mesh_names = cfg["mesh_names"] # ["Object_268", "Object_272", "Object_277"]
    all_points = []
    folder = cfg["foldermeshname"] #"booknshelvemesh"
    os.makedirs(folder, exist_ok=True)
    os.makedirs("franka_arm_meshes", exist_ok=True)


    # while True:
    #     env.step(np.array([0]))
    
    import torch
    import trimesh
    from omni.isaac.core.prims import RigidPrimView


    arm_mesh_prim_path = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 
    'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
    
    views  = []
    for prim_name in arm_mesh_prim_path:

        view = RigidPrimView(
                    prim_paths_expr=f"/World/envs/env_*/Robot/{prim_name}", reset_xform_properties=False
        )
        view.initialize()
        views.append(view)
    
    
    all_arm_points = []

    # i= 0
    # for prim_name in arm_mesh_prim_path:
    #     print(prim_name)
    #     mesh_prim_path = f'/World/envs/env_0/Robot/{prim_name}/visuals/{prim_name}'  # leaf level prim
    #     mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
    #     points_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'points'))
    #     faces_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'faceVertexIndices')).reshape(-1, 3)
    #     scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'xformOp:scale'))
    #     # if "184" in mesh_prim_path:
    #     points_scaled_np = points_np * scale_np

    #     tmesh = trimesh.Trimesh(vertices=points_scaled_np, faces=faces_np)

    #     points = tmesh.sample(5000)

    #     np.save(f"franka_arm_meshes/{prim_name}_pcd.npy", points)

    #     # from pxr import UsdGeom
    #     # import torch
    #     # parent_mesh_prim_path = f'/World/envs/env_0/Robot/{prim_name}'  # leaf level prim
    #     # parent_mesh_prim = prim_utils.get_prim_at_path(parent_mesh_prim_path)
    #     # xformable = UsdGeom.Xformable(parent_mesh_prim)
    #     # transform_matrix = xformable.GetLocalTransformation()


    #     pose = views[i].get_world_poses()
    #     pos = pose[0]
    #     rot = pose[1]
    #     print(pos, rot)
    #     # expand_points = np.concatenate([points, np.ones((len(points),1))], axis=1)
    #     # points_inv = (expand_points@np.array(transform_matrix))[:,:3]
    #     # data_name = prim_name.split("/")[-1]

    #     rotated_points = quat_apply( rot, torch.tensor(points).to("cuda").float())
    #     trans_points = rotated_points + pos
    #     all_arm_points.append(trans_points.cpu().detach().numpy())
    #     i+=1
    #     # all_arm_points.append(points_inv)
    

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_arm_points))
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(np.concatenate(all_points2)+0.01)
    # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(np.concatenate(all_points2)))
    o3d.visualization.draw_geometries([pcd])

    for prim_name in mesh_prim_paths:
        mesh_prim_path = f'/World/envs/env_0/Scene/{prim_name}/Geometry/Object_Geometry'  # leaf level prim
        mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
        points_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'points'))
        faces_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(mesh_prim), 'faceVertexIndices')).reshape(-1, 3)
        # root_prim = prim_utils.get_prim_at_path(f'/World/envs/env_0/Scene/Xform_184/Object_184/Geometry/Object_Geometry')
        # scale_np = np.asarray(prim_utils.get_prim_property(prim_utils.get_prim_path(root_prim), 'xformOp:scale'))
        scale = [1,1,1]
        if "184" in mesh_prim_path:
            scale = [3,3,3]
        points_scaled_np = points_np * np.array(scale)

        tmesh = trimesh.Trimesh(vertices=points_scaled_np, faces=faces_np)

        points = tmesh.sample(5000)
        

        from pxr import UsdGeom
        import torch
        xformable = UsdGeom.Xformable(mesh_prim)
        transform_matrix = xformable.GetLocalTransformation()

        expand_points = np.concatenate([points, np.ones((len(points),1))], axis=1)
        points_inv = (expand_points@np.array(transform_matrix))[:,:3]
        data_name = prim_name.split("/")[-1]
        
        if save:
            np.save(f"{folder}/{data_name}_pcd.npy", points_inv)
        
        
        data = env.base_env._env.env.scene._data
        
        object_pos = data[data_name].root_state_w[:,:3]
        object_rot = data[data_name].root_state_w[:,3:7]
        
        from scipy.spatial.transform import Rotation as R
        # rotation = R.from_quat(object_rot)
        # rotated_points = rotation.apply(points)
        rotated_points = quat_apply( object_rot, torch.tensor(points_inv).to("cuda").float())
        trans_points = rotated_points + object_pos
        all_arm_points.append(trans_points.cpu().detach().numpy())

    
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_arm_points))
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(np.concatenate(all_points2)+0.01)
    # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(np.concatenate(all_points2)))
    o3d.visualization.draw_geometries([pcd])


    ## Generate sample scene
    ## from state and joints

    # import IPython
    # IPython.embed()
    # while simulation_app.is_running():
    #     env.base_env._env.sim.step(True)
    # simulation_app.close()

def env_distance(env, state, goal):
        obs = env.observation(state)
        
        return env.compute_shaped_distance(obs, goal)
def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({"demos_video_trajectories":wandb.Video(images, fps=10)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=int, default=0)
    parser.add_argument("--seed",type=int, default=0)
    parser.add_argument("--env_name", type=str, default='pointmass_empty')
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
    parser.add_argument("--sensors",type=str, default=None)


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
    params.update(config["teleop"])
    params.update(config[args.distractors])
    params["render_images"] = True
    params["num_envs"] = 1
    
    params["sensors"] = ["synthetic_pcd"]
    for key in args.__dict__:
        value = args.__dict__[key]
        if value is not None:
            params[key] = value


    params["data_folder"] = data_folder_name
    del params["camera_rot"]
    wandb.init(project=args.env_name+"generate_meshes", config=params, dir="/data/pulkitag/data/marcel/wandb")


    run(**params)
    # dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')