import math
import numpy as np
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.render_product import set_resolution
import omni.replicator.isaac as dr
import omni.replicator.core as rep
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
from math import floor
import cv2
import open3d as o3d
import torch
import time
from camera_utils import project_depth_to_worldspace, compute_intrinsic_matrix, create_pointcloud_from_depth
from omni.isaac.core.prims import XFormPrim

MAX_NUM_CAMERAS = 23
MAX_NUM_CAMERAS = 250


class IsaacEnvRender:
    def __init__(self, id, 
                 task, 
                 first_person=True,
                 sensor: list = ["rgb"],
                 width: int = 1280, 
                 height: int = 960, 
                 fov: int = 62, # 71 
                 near: float = 0.10,
                 num_cameras: int = 2, 
                 far: float = 10000.0,
                 cfg=None):
        """
        Args:
            id: The id of the camera
            cam_pos: Location of camera
            cam_focus_pt: Location of camera target
            sensor: sensor type (rgb, depth, instanceSegmentation, semanticSegmentation, 
                       boundingBox2DTight, boundingBox2DLoose, boundingBox3D, camera)
            width: The horizontal image resolution in pixels
            height: The vertical image resolution in pixels
            fov: The field of view of the camera
            near: The near plane distance
            far: The far plane distance
        """
        self.id = id
        self.num_cameras = num_cameras
        # self.cam_rot = camera_rot
        self.camera_pos_rand = cfg.get("camera_pos_rand")
        self.cam_pos = cfg.get("camera_pos")
        self.camera_target_rand = cfg.get("camera_target_rand")
        self.camera_target = cfg.get("camera_target")
        self.camera_rot = cfg.get("camera_rot")
        # if self.num_cameras == 1:
        #     self.cam_pos = np.array([[float(x) for x in camera_pos.split(",")]])
        #     self.cam_rot = np.array([[float(x) for x in camera_rot.split(",")]])
        # else:
        #     self.cam_pos = np.array([
        #         [-2.1397,0.31284,1.87657],
        #         [0.29774,-1.65697,0.73069],
        #         [0.4868,1.92378,0.68305],
        #         [0.35839,0.21813,2.80389],
        #     ])
        #     self.cam_rot = np.array([
        #         [-0.57102,-0.35844,0.39266,0.62552],
        #         [0.76995,0.63741,-0.019,-0.02294],
        #         [-0.02434,-0.02242,0.67721,0.73504],
        #         [0.70858,0.0198,-0.01971,-0.70507],
        #     ])
        # self.cam_focus_pt = np.array(cam_focus_pt)
        self.sensor = sensor
        self._width = width
        self._height = height
        self.__fov = fov
        self.__near = near
        self.__far = far
        self.__aspect = self._width / self._height
        self._view_matrix = None
        self.first_person = first_person
        self.device = "cuda"

        self.num_envs = min(task.num_envs, MAX_NUM_CAMERAS)
        # print("environment", task.env_ns, task.template_env_ns)
        self.task_env_ns = task.env_ns
        if self.first_person:
            self._env_prim_paths = [f"{task.env_ns}/env_0/Robot/panda_hand"]#env_ns#_env_prim_paths
        else:
            self._env_prim_paths = [f"{task.env_ns}/env_0"]#env_ns#_env_prim_paths

        self._np_float64_env_pos = task.envs_positions.cpu().numpy()#_np_float64_env_pos

        # print("env pos", self._np_float64_env_pos)
        fov_horizontal = self.__aspect * self.__fov 
        fov_vertical = self.__fov



        # fov_horizontal = 86
        # fov_vertical = 57
        if cfg["cam_idx"] == 2:
            # TODO: find the right camera parameters
            focal_length = 1.88 # check parameters online
            attributes = {"horizontalAperture": 2*focal_length*math.tan(fov_horizontal*math.pi/180/2), 
                        "verticalAperture": 2*focal_length*math.tan(fov_vertical*math.pi/180/2),
                        "focalLength": focal_length,
                        "clippingRange": (self.__near, self.__far)
                        }
        else:
            intrinsics_gt = np.array(
                [[613.14752197,   0.        , 326.19647217],
                [  0.        , 613.16229248, 244.59855652],
                [  0.        ,   0.        ,   1.        ]])
            fx_gt = intrinsics_gt[0, 0]
            cx_gt = intrinsics_gt[0, 2]
            fy_gt = intrinsics_gt[1, 1]
            cy_gt = intrinsics_gt[1, 2]

            focal_length = 1.0 # ??
            # self.width = 326.19647217 * 2
            # self.height = 244.59855652 * 2
            width, height = float(self._width), float(self._height)
            attributes = {"horizontalAperture": 0.5 * width * focal_length / fx_gt, 
                        "verticalAperture": 0.5 * height * focal_length / fy_gt,
                        "focalLength": focal_length,
                        "clippingRange": (self.__near, self.__far),
                        #   "horizontalApertureOffset": (cx_gt - width / 2) / fx_gt,
                        #   "verticalApertureOffset": (cy_gt - height / 2) / fy_gt
                        }
        print("Attributes", attributes)

        # self.viewport = get_active_viewport()
        self.annotators = self._create_annotators(sensors=self.sensor)
        self._create_camera(attributes)
        # if not self.first_person:
        #     self.set_camera_pose(cam_pos=self.cam_pos,
        #                         cam_focus_pt=self.cam_focus_pt)


    def reset(self):

        camera_state_pos = torch.tensor([self.cam_pos.copy() for i in range(self.num_envs)]).to(self.device)
        bound = torch.tensor(self.camera_pos_rand).to(self.device)
        camera_state_pos[:] += sample_uniform(
                -bound, bound, (self.num_envs,self.num_cameras, 3), device=self.device
            )

        for i in range(min(self.num_envs,MAX_NUM_CAMERAS)):
            for j in range(self.num_cameras):
                cam_prim_path = self.camera_prim_paths[i*self.num_cameras+j]
                if self.camera_rot and len(self.camera_rot) != 0:
                    xform = XFormPrim(cam_prim_path)
                    
                    # camera_state_rot = torch.tensor([self.cam_rot.copy() for i in range(self.num_envs)]).to(self.device)

                    # rand_rot = self.random_orientation_with_bounds(self.num_envs)
                    # camera_state_rot[:] = quat_mul(rand_rot, camera_state_rot[:, :self.num_cameras])

                    ori = self.camera_rot[j]
                    xform.set_local_pose(camera_state_pos[i,j].cpu().numpy(), np.array(ori))
                else:
                    camera_target_pos = torch.tensor(np.array([self.camera_target.copy() for i in range(self.num_envs*self.num_cameras)]).reshape(self.num_envs,self.num_cameras,-1)).to(self.device).type(torch.float)
                    bound = torch.tensor(self.camera_target_rand).to(self.device)
                    camera_target_pos[:] += sample_uniform(
                            -bound, bound, (self.num_envs,self.num_cameras, 3), device=self.device
                    )
                    # cam_pose = compute_cam_pose(camera_state_pos[i,j].cpu().numpy())
                    # print("camera pose", compute_cam_pose(2*self._np_float64_env_pos[i] + camera_state_pos[i,j].cpu().numpy()))
                    set_camera_view(2*self._np_float64_env_pos[i] + camera_state_pos[i,j].cpu().numpy(), camera_target_pos[i,j].cpu().numpy()+self._np_float64_env_pos[i]*2, camera_prim_path=cam_prim_path)
                
    
    # def random_orientation_with_bounds(self, num: int) -> torch.Tensor:
    #     """Returns sampled rotation around z-axis.

    #     Args:
    #         num (int): The number of rotations to sample.
    #         device (str): Device to create tensor on.

    #     Returns:
    #         torch.Tensor: Sampled quaternion (w, x, y, z).
            
    #     """

    #     bound = torch.tensor([self.camera_rot_rand.copy() for i in range(self.num_envs)]).to(self.device)
    #     import IPython
    #     IPython.embed()
    #     roll = (2*bound[:,0].reshape(-1, self.num_cameras)) * torch.rand((self.num_envs,self.num_cameras), dtype=torch.float, device=self.device) - bound[:,0]
    #     pitch = (2*bound[:,1]) * torch.rand((self.num_envs,self.num_cameras), dtype=torch.float, device=self.device) - bound[:,1]
    #     yaw = (2*bound[:,2]) * torch.rand((self.num_envs,self.num_cameras), dtype=torch.float, device=self.device) - bound[:,2]

    #     return quat_from_euler_xyz(roll, pitch, yaw)

    def render(self, sensors=["rgb"]):
        render_products = self._create_annotators(sensors)
        render_products["depth_img"] = []
        start = time.time()
        rep.orchestrator.step(pause_timeline=False)
        print("render took", time.time() - start)
        # import IPython
        # IPython.embed()
        for sensor in sensors:
            for idx, annot in enumerate(self.annotators[sensor]):
                data = annot.get_data()
                # import IPython
                # IPython.embed()
                if "segmentation" in sensor:
                    data = data["data"]

                if "pointcloud" == sensor:
                    pcd_points = data["data"]
                    pcd_colors = data["info"]["pointRgb"].reshape(-1,4)[:,:3]
                    
                    render_products[sensor].append((pcd_points, pcd_colors))
                    # o3d.visualization.draw_geometries([pcd])
                elif "distance_to_image_plane" == sensor:
                    prim_path = f"/World/envs/env_{idx}/camera_0"
                    render_products["depth_img"].append(data)
                    print(f'Here calling "create_pointcloud_from_depth"')
                    points = create_pointcloud_from_depth(prim_path, data, device="cuda")
                    
                    render_products[sensor].append((points, np.ones_like(points)))
                else:
                    image_data = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
                    render_products[sensor].append(image_data)
        
        if self.num_cameras == 2:
            key = "rgb"
            images = render_products[key]
            
            render_products[key] = []

            for i in range(min(4, self.num_envs)):
                extra_pixels = 40
                new_image = np.concatenate([
                    np.concatenate([images[i*2+0], np.zeros((extra_pixels,images[0].shape[1], 4,1)), images[i*2+1]], axis=0), 
                    np.zeros((extra_pixels + images[1].shape[0]*2, extra_pixels, 4,1)), 
                    np.concatenate([np.zeros_like(images[0]), np.zeros((extra_pixels, images[0].shape[1], 4,1)), np.zeros_like(images[0])], axis=0),
                    np.ones((extra_pixels + images[1].shape[0]*2, extra_pixels, 4,1))*255, 
                ], axis=1).reshape(self._height+extra_pixels, self._width+extra_pixels*2, 4)
                render_products[key].append((np.floor(cv2.resize(new_image, (self._height, self._width)).reshape(self._height, self._width, 4, 1))).astype(np.uint8))


        if self.num_cameras == 4:
            key = "rgb"
            images = render_products[key]
            extra_pixels = 40
            render_products[key] = np.concatenate([
                np.concatenate([images[0], np.zeros((extra_pixels,images[0].shape[1], 4,1)), images[1]], axis=0), 
                np.zeros((extra_pixels + images[1].shape[0]*2, extra_pixels, 4,1)), 
                np.concatenate([images[2], np.zeros((extra_pixels, images[2].shape[1], 4,1)), images[3]], axis=0),
                np.ones((extra_pixels + images[1].shape[0]*2, extra_pixels, 4,1))*255, 
            ], axis=1).reshape(self._height+extra_pixels, self._width+extra_pixels*2, 4)
            render_products[key] = (np.floor(cv2.resize(render_products[key], (self._height, self._width)).reshape(1, self._height, self._width, 4, 1))).astype(np.uint8)

        # pcd_depth = render_products["distance_to_image_plane"][9]
        # pcd = render_products["pointcloud"][9]
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(pcd_depth[0])
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(pcd[0])
        # o3d.visualization.draw_geometries([pcd1,pcd2])
        return render_products

    # def set_camera_pose(self, cam_pos=[-1,-1.7,1.4], cam_focus_pt=[0,0,0]):
    #     for i in range(self.num_envs):
    #         for j in range(self.num_cameras):
    #             camera_prim_path = f"{self.task_env_ns}/env_{i}/camera_{j}"
    #             set_camera_view(eye=self.cam_pos[j], 
    #                             target=cam_focus_pt, 
    #                             camera_prim_path=camera_prim_path)
    #         #prim_utils.set_prim_property(camera_prim_path, "translation", [0,0,0])

    # TODO SET RESOLUTION (640 x 480) downsample to half
    # change background
    # change image to uint8
    # check the data augmentation
    # brightnes, contrast, randomize the lighting in the environment
    # randomize camera position
    # focal point should change
    # randomize at the beginning of the episodes
    # check control frequncy
    # check adding the camera to the body of the franka arm
    # end effector control + inverse kinematics + delta step (add canmera to the end effector)
    def set_resolution(self, width, height):

        self._width = width
        self._height = height
        # self.width = int(326.19647217 * 2)
        # self.height = int(244.59855652 * 2)

        for rp in self.replicators:
            if self.num_cameras == 1:
                set_resolution(rp, (floor(width), floor(height)))
            else:
                set_resolution(rp, (floor(width//2), floor(height//2)))


    def _create_camera(self, attributes):
        self.camera_prim_paths = []
        self.camera_prims = []
        self.replicators = []
        self.viewports = []
        # for i in range(len(self._env_prim_paths)):

        for i in range(self.num_cameras):
            camera_prim_path = f"{self._env_prim_paths[0]}/camera_{i}"
            print("camera prim path third person", camera_prim_path)
            if self.first_person:
                camera_prim = create_prim(prim_path=camera_prim_path, prim_type="Camera", translation=[0.13,0,-0.05], orientation=[0.16507,0.68757,0.68757,0.16507], attributes=attributes)
            else:
                camera_prim = create_prim(prim_path=camera_prim_path, prim_type="Camera", translation=self.cam_pos[i], orientation=[0.16507,0.68757,0.68757,0.16507], attributes=attributes)
                # camera_prim = create_prim(prim_path=camera_prim_path, prim_type="Camera", translation=[-0.04748,2.07405,1.70903], orientation=[-0.01034,-0.00522,0.45096,0.89247], attributes=attributes)
            

            
        for i in range(self.num_envs):
            if i >= (MAX_NUM_CAMERAS+1)/self.num_cameras:
                break
            for j in range(self.num_cameras):
                if self.first_person:
                    camera_prim_path = f"{self.task_env_ns}/env_{i}/Robot/panda_hand/camera_{j}"
                else:
                    camera_prim_path = f"{self.task_env_ns}/env_{i}/camera_{j}"
                rp = rep.create.render_product(camera_prim_path, resolution=(self._width, self._height))
                self.camera_prim_paths.append(camera_prim_path)
                self.camera_prims.append(camera_prim)

                self.replicators.append(rp)

                for sensor in self.sensor:
                    annot = rep.AnnotatorRegistry.get_annotator(sensor).attach([rp])
                    self.annotators[sensor].append(annot)

    def _create_annotators(self, sensors):
        annotators = {}
        for sensor in sensors:
            annotators[sensor] = []
        return annotators