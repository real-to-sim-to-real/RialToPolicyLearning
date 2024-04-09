import time
from utils import preprocess_pcd,preprocess_pcd_from_canon, postprocess_pcd, unpad_points
import numpy as np
import wandb

def rollout_policy_real(env, policy, urdf, cfg, render=True, from_state=True, expert_policy=None):
    policy.eval()

    actions = []
    states = []
    joints = []
    cont_actions = []
    all_pcds_points = []
    all_pcds_colors = []
    all_pcds_points_full = []
    all_pcds_colors_full = []
    expert_actions = []
    images = []

    debug = True
    state = env.reset()
    joint = env.get_robot_joints()
    start_demo =time.time()
    for t in range(cfg['max_path_length']):
        
        start_timestep = time.time()
        if render:
            start = time.time()
            img, pcd = env.render_image(cfg["sensors"])
            print("Rendering pcd image", time.time()-start)

            points, colors = pcd
            points = points.reshape((1, -1, 3))
            colors = colors.reshape((1, -1, 3))
            start_prc = time.time()
            # check last joint
            if cfg["presample_arm_pcd"]:
                # this is faster (doesn't re-sample the points from the mesh each time)
                pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
            else:
                pcd_processed_points = preprocess_pcd(pcd, joint, urdf, cfg)
            print("End proc ", time.time() -  start_prc)
            start_prc = time.time()
            pcd_processed_points_full, pcd_processed_colors_full = postprocess_pcd(pcd_processed_points, cfg) 
            print("end proc 2", time.time() - start_prc)
            print("state", state)
            # pcd_processed_points_full, pcd_processed_colors_full, pcd_processed_points, pcd_processed_colors = process_pcd((points, colors), joint, urdf, cfg)
            # import IPython
            # IPython.embed()
            if t == 0:
                import open3d as o3d
                pcd1 = o3d.geometry.PointCloud()
                pcd1.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points_full[0]))
                o3d.visualization.draw_geometries([pcd1])

                vis = o3d.visualization.Visualizer()
                vis.create_window()
                vis.add_geometry(pcd1)
            
            vis.remove_geometry(pcd1)

            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points_full[0]))
            vis.add_geometry(pcd1)
            vis.poll_events()
            vis.update_renderer()

        if from_state:
            observation = env.observation(state)

            action = policy.act_vectorized(observation, observation)
        else:

            state = state.reshape(1, -1)
            if cfg["rnn"]:
                with torch.no_grad():
                    pcd_processed_points_full_par, pcd_processed_colors_full_par, state_par = np.expand_dims(pcd_processed_points_full, axis=1), np.expand_dims(pcd_processed_colors_full, axis=1), state.unsqueeze(1)
                    action = policy((pcd_processed_points_full_par, pcd_processed_colors_full_par, state_par), init_belief=(t==0)).detach().squeeze().argmax(dim=1).cpu().numpy()    

            else:
                action = policy((pcd_processed_points_full, pcd_processed_colors_full, state)).argmax(dim=1).cpu().numpy()

        if expert_policy:
            expert_action = expert_policy.act_vectorized(observation, observation)
            expert_actions.append(expert_action)

        if render:
            all_pcds_points.append(pcd_processed_points)
            all_pcds_colors.append([])
            all_pcds_points_full.append(pcd_processed_points_full)
            all_pcds_colors_full.append(pcd_processed_colors_full)
            images.append(img) 

        if expert_policy and np.random.random() < cfg["sampling_expert"] :
            action = expert_action

        actions.append(action)
        states.append(state)
        joints.append(joint.detach().cpu().numpy())
        print("Forward pass", time.time() - start_timestep)
        state, _, done , info = env.step(action)
        joint = info["robot_joints"]

        if cfg["reset_if_open"] and action == 12:
            print("resetting because open")
            break

        print("sum actions open", np.sum(np.array(actions) == 12))
        if action == 12 and np.sum(np.array(actions) == 12) > cfg["nb_open"]:
            print("resetting because too ,many open")
            break

        # cont_action = info["cont_action"].detach().cpu().numpy()
        # cont_actions.append(cont_action)

    # success = env.base_env._env.get_success().detach().cpu().numpy()
    print(f"Trajectory took {time.time() - start_demo}")

    actions = np.array(actions).transpose(1,0)
    # cont_actions = np.array(cont_actions).transpose(1,0,2)
    # states = np.array(states).transpose(1,0,2)
    # joints = np.array(joints).transpose(1,0,2)

    if expert_policy:
        expert_actions = np.array(expert_actions).transpose(1,0)
    else:
        expert_actions = None

    if render:
        all_pcd_points = np.array(all_pcds_points).transpose(1,0,2,3) # TODO TRANSPOSE
        all_pcd_colors = np.ones_like(all_pcd_points)
        all_pcd_colors = all_pcd_colors[...,0]
        all_pcd_colors = all_pcd_colors[...,None]
        all_pcd_points_full = np.array(all_pcds_points_full).transpose(1,0,2,3) # TODO TRANSPOSE
        all_pcd_colors_full = np.array(all_pcds_colors_full).transpose(1,0,2,3) # TODO TRANSPOSE
        images = np.array(images).transpose(1,0,2,3,4)
    success = np.array([False])
    wandb.log({"Success": np.mean(success), "Time/Trajectory": time.time() - start_demo})
    return actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success
