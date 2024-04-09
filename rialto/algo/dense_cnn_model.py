
import copy
import numpy as np
import torch
from torch import nn

from rialto.algo.dense_cnn.scene_encoder import LocalPoolPointnet
import meshcat
try:
    from torchsparse import nn as spnn
except ImportError as e:
    print(f'Could not import from torchsparse - sparse models unavailable ({e})')


pt_dim = 3
padding = 0.1
voxel_reso_grid = 32
voxel_reso_grid_pt = 128

class DenseConvPolicy(nn.Module):
    def __init__(
            self,
            in_channels,
            # obs_size,
            act_size,
            augment_obs=False,
            augment_points=False,
            rgb_feats=False,
            pad_points=False,
            layers=[256,256],
            emb_layers=[64],
            pcd_normalization=None,
            pcd_scene_scale=1.0,
            emb_size=128,
            dropout = 0,
            nonlinearity=torch.nn.ReLU,
            use_state=False,
            state_dim=9, # tool position, tool orientation, open 3 + 4 + 2
            encoder_type="resnet",
            pool="avg",
            scene_encoder_kwargs = None,
            device="cuda"
    ):

        super(DenseConvPolicy, self).__init__()

        self.emb_size = emb_size
        self.use_state = use_state
        self.device = device

        self.scene_encoder_kwargs = scene_encoder_kwargs
        
        self.scene_offset = torch.Tensor([[pcd_normalization]]).float().to(self.device)
        self.scene_scale = pcd_scene_scale

        # mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        # mc_vis['scene'].delete()
        # Encoder
        self.plane_type = self.scene_encoder_kwargs["plane_type"]
        self.point_encoder = LocalPoolPointnet(
            dim=pt_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=None,
            **self.scene_encoder_kwargs).cuda()


        net_layers = []
        dim = 0
        for key in self.plane_type:
            if key == "grid":
                dim += 128
            else:
                dim += 64
        
        if len(self.plane_type) == 4:
            layers[0] = 512

        if self.use_state:
            dim += state_dim
        for i, layer_size in enumerate(layers):
            net_layers.append(torch.nn.Linear(dim, layer_size))
            net_layers.append(nonlinearity())
            dim = layer_size
        if dropout > 0:
            net_layers.append(torch.nn.Dropout(dropout))
        dim = layer_size
        net_layers.append(torch.nn.Linear(dim, act_size))
        self.layers = net_layers
        self.mlp = torch.nn.Sequential(*net_layers).to("cuda")

    def forward(self, x):
        coords, feats, obs = x
        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd_demo = o3d.geometry.PointCloud()
        # pcd_demo.points = o3d.utility.Vector3dVector(np.concatenate(coords))
        # pcd_demo.colors = o3d.utility.Vector3dVector(np.zeros_like(coords[0]))
        # # demo_points = np.load("real_world_pcd.npy")
        # # pcd_current = o3d.geometry.PointCloud()
        # # pcd_current.points = o3d.utility.Vector3dVector(demo_points[0])
        # o3d.visualization.draw_geometries([pcd_demo])
        # # np.save("real_world_pcd.npy", coords)
        # # unpad points

        B, N, C = coords.shape

        coords = torch.tensor(coords).to(self.device).float()
        
        coords = (coords - self.scene_offset.repeat((B, N, 1))) * self.scene_scale  # normalize

        points_embed_all = self.point_encoder(coords)
        all_features = []
        for key in points_embed_all.keys():
            points_embed = points_embed_all[key]
            feat_dim = points_embed.shape[1] #self.scene_encoder_kwargs['unet3d_kwargs']['out_channels']
            if key == "grid":
                fea_grid = points_embed.permute(0, 2, 3, 4, 1)
                flat_fea_grid = fea_grid.reshape(B, -1, feat_dim)
                global_fea1_mean = flat_fea_grid.mean(1)
                global_fea1_max = flat_fea_grid.max(1).values
                all_features.append(global_fea1_max)
                all_features.append(global_fea1_mean)
            else:
                plane_embed = points_embed.permute(0, 2, 3, 1).reshape(B, -1, feat_dim)  # (B, 128**2, 32)
                plane_fea1_mean = plane_embed.mean(1)  # (B, 32x3) = (B, 96)
                plane_fea1_max = plane_embed.max(1).values  # (B, 32x3) = (B, 96)
                all_features.append(plane_fea1_max)
                all_features.append(plane_fea1_mean)
            
            # plane_fea_emb = torch.hstack([plane_fea1_mean, plane_fea1_max])  # (B, 96x2) = (B, 192)
            
        point_emb = torch.hstack(all_features)

        if self.use_state:
            mlp_input = torch.hstack([point_emb, obs])
        else:
            mlp_input = point_emb

        logits = self.mlp(mlp_input)
    
        return logits

    def compute_loss(self, coords, feats, obs, actions):

        logits = self.forward((coords, feats, obs))
        loss = nn.CrossEntropyLoss( reduction='mean')(logits, actions)
        return loss

class SparseFeatureExtractor(nn.Module):
    def __init__(
            self,
            obs_space,
            # obs_size,
            augment_obs=False,
            augment_points=False,
            rgb_feats=False,
            pad_points=False,
            layers=[256,256],
            emb_layers=[64],
            dropout = 0,
            nonlinearity=torch.nn.ReLU,
            state_dim=9, # tool position, tool orientation, open 3 + 4 + 2
            encoder_type="resnet",
            pool="avg",
            device="cuda"
    ):

        super(SparseFeatureExtractor, self).__init__()

        self.device = device

        self.features_dim = 96*2

        self.scene_encoder_kwargs = {
            'local_coord': True,
            # 'encoder': pointnet_local_pool,
            'c_dim': 32,
            # 'encoder_kwargs':
            'hidden_dim': 32,
            # 'plane_type': ['xz', 'xy', 'yz', 'grid'],
            'plane_type': ['grid_sparse'],
            # 'grid_resolution': 32,
            'unet3d': True,
            'unet3d_kwargs': {
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 64, #32,
                'plane_resolution': 128,
            },
            'unet': False,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 32
            },
            'sparse': True,
        }


        # mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        # mc_vis['scene'].delete()
        # Encoder
        self.point_encoder = LocalPoolPointnet(
            dim=pt_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=None,
            **self.scene_encoder_kwargs).cuda()

        self.avg_pooling_layer = spnn.GlobalAvgPool()
        self.max_pooling_layer = spnn.GlobalMaxPool()

    def forward(self, x):
        # coords, feats, obs = x
        coords = x.float()
        # import IPython
        # IPython.embed()
        # # import open3d as o3d
        # pcd_demo = o3d.geometry.PointCloud()
        # pcd_demo.points = o3d.utility.Vector3dVector(coords[0].cpu().numpy())
        # pcd_demo.colors = o3d.utility.Vector3dVector(np.zeros_like(coords[0].cpu().numpy()))
        # demo_points = np.load("real_world_pcd.npy")
        # pcd_current = o3d.geometry.PointCloud()
        # pcd_current.points = o3d.utility.Vector3dVector(demo_points[0])
        # o3d.visualization.draw_geometries([pcd_demo])
        # np.save("real_world_pcd.npy", coords)
        # unpad points

        B, N, C = coords.shape

        # coords = torch.tensor(coords).to(self.device).float()
        points_embed = self.point_encoder(coords)["grid_sparse"]
        feat_dim = self.scene_encoder_kwargs['unet3d_kwargs']['out_channels']
        
        global_fea1_mean = self.avg_pooling_layer(points_embed)
        global_fea1_max = self.max_pooling_layer(points_embed)
        point_emb = torch.hstack([global_fea1_max, global_fea1_mean])
       
        return point_emb

 

class DenseFeatureExtractor(nn.Module):
    def __init__(
            self,
            obs_space,
            # obs_size,
            augment_obs=False,
            augment_points=False,
            rgb_feats=False,
            pad_points=False,
            layers=[256,256],
            emb_layers=[64],
            dropout = 0,
            nonlinearity=torch.nn.ReLU,
            state_dim=9, # tool position, tool orientation, open 3 + 4 + 2
            encoder_type="resnet",
            pool="avg",
            device="cuda"
    ):

        super(DenseFeatureExtractor, self).__init__()

        self.device = device

        self.features_dim = 128

        self.scene_encoder_kwargs = {
            'local_coord': True,
            # 'encoder': pointnet_local_pool,
            'c_dim': 32,
            # 'encoder_kwargs':
            'hidden_dim': 32,
            # 'plane_type': ['xz', 'xy', 'yz', 'grid'],
            'plane_type': ['grid'],
            # 'grid_resolution': 32,
            'unet3d': True,
            'unet3d_kwargs': {
                'num_levels': 3,
                'f_maps': 32,
                'in_channels': 32,
                'out_channels': 64, #32,
                'plane_resolution': 128,
            },
            'unet': False,
            'unet_kwargs': {
                'depth': 5,
                'merge_mode': 'concat',
                'start_filts': 32
            }
        }


        # mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        # mc_vis['scene'].delete()
        # Encoder
        self.point_encoder = LocalPoolPointnet(
            dim=pt_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=None,
            **self.scene_encoder_kwargs).cuda()



    def forward(self, x):
        # coords, feats, obs = x
        coords = x.float()
        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd_demo = o3d.geometry.PointCloud()
        # pcd_demo.points = o3d.utility.Vector3dVector(coords[0].cpu().numpy())
        # pcd_demo.colors = o3d.utility.Vector3dVector(np.zeros_like(coords[0].cpu().numpy()))
        # demo_points = np.load("real_world_pcd.npy")
        # pcd_current = o3d.geometry.PointCloud()
        # pcd_current.points = o3d.utility.Vector3dVector(demo_points[0])
        # o3d.visualization.draw_geometries([pcd_demo])
        # np.save("real_world_pcd.npy", coords)
        # unpad points

        B, N, C = coords.shape

        # coords = torch.tensor(coords).to(self.device).float()
        points_embed = self.point_encoder(coords)["grid"]
        feat_dim = self.scene_encoder_kwargs['unet3d_kwargs']['out_channels']

        fea_grid = points_embed.permute(0, 2, 3, 4, 1)
        flat_fea_grid = fea_grid.reshape(B, -1, feat_dim)
        global_fea1_mean = flat_fea_grid.mean(1)
        global_fea1_max = flat_fea_grid.max(1).values
        point_emb = torch.hstack([global_fea1_max, global_fea1_mean])
    
        return point_emb

 


        
if __name__ == "__main__":
    from util import three_util, util
    import meshcat
    import trimesh


    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    mc_vis['scene'].delete()
    # points = np.random.random((10000, 3))
    import IPython
    IPython.embed()
    points = trimesh.creation.uv_sphere(1.0).sample(10000)
    points2 = (np.random.random((100, 3)) - 0.5) * 0.5

    points = np.load("/scratch/marcel/data/online_distill/demos/booknshelvesyntheticonline/pcd_points_8_4997.npy")[0,0]
    points2 = np.load("/scratch/marcel/data/online_distill/demos/booknshelvesyntheticonline/pcd_points_8_4997.npy")[0,100]
    mc_size = 0.005
    util.meshcat_pcd_show(mc_vis, points, (255, 0, 0), name='scene/points', size=mc_size)
    util.meshcat_pcd_show(mc_vis, points2, (0, 0, 255), name='scene/points2', size=mc_size)

    points_bb = trimesh.PointCloud(points).bounding_box
    points_mean = copy.deepcopy(np.asarray(points_bb.centroid))
    points_extents = copy.deepcopy(np.asarray(points_bb.extents))
    points_scale = 1 / np.max(points_extents)

    # points_norm = (points - points_mean) * points_scale
    # points2_norm = (points2 - points_mean) * points_scale

    # util.meshcat_pcd_show(mc_vis, points_norm, (255, 0, 0), name='scene/points_norm', size=mc_size)
    # util.meshcat_pcd_show(mc_vis, points2_norm, (0, 0, 255), name='scene/points2_norm', size=mc_size)

    points = torch.from_numpy(points).float().cuda().reshape(1, -1, 3)
    points2 = torch.from_numpy(points2).float().cuda().reshape(1, -1, 3)

    points_full = torch.cat((points, points2), dim=1)

    B = 1
    feat_dim = scene_encoder_kwargs['unet3d_kwargs']['out_channels']

    fea_grid = scene_encoder(points_full, debug_viz=True)
    fea_grid = fea_grid['grid'].permute(0, 2, 3, 4, 1)
    flat_fea_grid = fea_grid.reshape(B, -1, feat_dim)
            
    pts2_mean_raster_index = three_util.coordinate2index(
        three_util.normalize_3d_coordinate(torch.mean(points2[:, :, :3], dim=1).reshape(B, -1, 3)),
        voxel_reso_grid,
        '3d').squeeze()

    pts2_raster_index = three_util.coordinate2index(
        three_util.normalize_3d_coordinate(points2[:, :, :3].reshape(B, -1, 3)),
        voxel_reso_grid,
        '3d').squeeze()

    pts_raster_index = three_util.coordinate2index(
        three_util.normalize_3d_coordinate(points[:, :, :3].reshape(B, -1, 3)),
        voxel_reso_grid,
        '3d').squeeze()

    pts_full_raster_index = three_util.coordinate2index(
        three_util.normalize_3d_coordinate(points_full[:, :, :3].reshape(B, -1, 3)),
        voxel_reso_grid,
        '3d').squeeze()
            
    global_fea1 = flat_fea_grid.mean(1)  # mean feature across the whole grid (could be max)
    global_fea2 = flat_fea_grid.gather(dim=1, index=pts_full_raster_index.reshape(B, -1, 1).repeat((1, 1, feat_dim))).mean(1)  # mean feature across voxels with points
    global_fea3 = flat_fea_grid.gather(dim=1, index=pts2_raster_index.reshape(B, -1, 1).repeat((1, 1, feat_dim))).mean(1)  # mean feature across points in a specific region 
    global_fea4 = flat_fea_grid.gather(dim=1, index=pts2_mean_raster_index.reshape(B, -1, 1).repeat((1, 1, feat_dim))).reshape(B, -1)  # feature at a specific voxel

    # 3D spatial softmax?

    from IPython import embed; embed()
