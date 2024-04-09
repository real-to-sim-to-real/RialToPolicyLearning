import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
import meshcat
import trimesh
import time

from rialto.algo.dense_cnn.encoder.layers import ResnetBlockFC
from rialto.algo.dense_cnn.encoder.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
from rialto.algo.dense_cnn.encoder.common import make_3d_grid
from rialto.algo.dense_cnn.encoder.unet import UNet
from rialto.algo.dense_cnn.encoder.unet3d import UNet3D
from rialto.algo.dense_cnn.util import util, three_util
try:
    from torchsparse import SparseTensor
    from torchsparse.utils.collate import sparse_collate
except ImportError as e:
    print(f'Could not improt from torchsparse - sparse models unavailable ({e})')

class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, 
                 local_coord=False, pos_encoding='linear', mc_vis=None, sparse=False):
        super().__init__()
        self.mc_vis = mc_vis
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            if sparse:
                from torchsparse.backbones import SparseResUNet42
                self.unet3d = SparseResUNet42(in_channels=32)
            else:
                self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        if local_coord:
            unit_size = 1.1 / self.reso_grid 
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2*hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2*hidden_dim)


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features_sparse(self, index, c):
        # scatter grid features from points        
        # c = c.permute(0, 2, 1)
        # if index.max() < self.reso_grid**3:
        #     fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
        #     fea_grid = scatter_mean(c, index, out=fea_grid) # B x c_dim x reso^3
        # else:
        #     fea_grid = scatter_mean(c, index) # B x c_dim x reso^3
        #     if fea_grid.shape[-1] > self.reso_grid**3: # deal with outliers
        #         fea_grid = fea_grid[:, :, :-1]
        # fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        sparse = []
        for coord, feat in zip(index['coord_grid_sparse'], c):
            
            # coord = torch.tensor(coord, dtype=torch.int8).to("cuda")
            # feat = torch.tensor(feat, dtype=torch.float).to("cuda")
            coord = coord.int()
            tensor = SparseTensor(coords=coord, feats=feat)
            sparse.append(tensor)

        # combine and add batch dimension
        fea_grid = sparse_collate(sparse)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid[-1]

    def generate_grid_features(self, index, c):
        # scatter grid features from points        
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid**3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
            fea_grid = scatter_mean(c, index, out=fea_grid) # B x c_dim x reso^3
        else:
            fea_grid = scatter_mean(c, index) # B x c_dim x reso^3
            if fea_grid.shape[-1] > self.reso_grid**3: # deal with outliers
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = index.keys()

        c_out = 0
        for key in keys:
            if key == "coord_grid_sparse":
                continue
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            elif key == 'grid_sparse':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p, debug_viz=False):
        batch_size, T, D = p.size()

        start = time.time()

        p_coords = p[:, :, :3]

        # acquire the index for each point
        coord = {}
        index = {}

        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p_coords.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p_coords.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p_coords.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p_coords.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')
        if 'grid_sparse' in self.plane_type:
            coord['grid_sparse'] = normalize_3d_coordinate(p_coords.clone(), padding=self.padding)
            index['coord_grid_sparse'] = (coord['grid_sparse']*self.reso_grid).long() #coordinate2index(coord['grid_sparse'], self.reso_grid, coord_type='3d')
            index['grid_sparse'] = coordinate2index(coord['grid_sparse'], self.reso_grid, coord_type='3d')
        # print('Here in pointnet local forward, after getting index and coord')
        # from IPython import embed; embed()

        if self.map2local:
            pp = self.map2local(p_coords)  # only map first three coords to local (others might be non-coords)
            if p.shape[1] > 3:
                pp = torch.cat((pp, p[:, :, 3:]), dim=-1)

            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)

        if debug_viz:
            # get the B x N x 3 raster points
            raster_pts = make_3d_grid((-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), (self.reso_grid, self.reso_grid, self.reso_grid))[None, :, :].repeat(batch_size, 1, 1).cuda() #.numpy()

            # reshape to grid, and swap axes (permute x and z), B x reso x reso x reso x 3
            raster_pts = raster_pts.reshape(batch_size, self.reso_grid, self.reso_grid, self.reso_grid, 3)
            raster_pts = raster_pts.permute(0, 3, 2, 1, 4)

            # reshape back to B x N x 3
            raster_pts = raster_pts.reshape(batch_size, -1, 3)
            filled_raster_pts = raster_pts.permute(0, 2, 1).gather(dim=2, index=index['grid'].expand(-1, 3, -1)).permute(0, 2, 1)

            filled_raster_pts_np = filled_raster_pts[0].detach().cpu().numpy()
            coord_np = p[0].detach().cpu().numpy().squeeze()

        if debug_viz and False:
            util.meshcat_pcd_show(self.mc_vis, filled_raster_pts_np, (0, 255, 0), 'scene/filled_raster')
            util.meshcat_pcd_show(self.mc_vis, coord_np, (255, 0, 0), 'scene/point_cloud')

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            # pooled = self.pool_local(coord, index, net)
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        # print('Here in pointnet local forward, after fc_c')
        # from IPython import embed; embed()
        
        if debug_viz and False:
            raster_pts = make_3d_grid((-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), (self.reso_grid, self.reso_grid, self.reso_grid))[None, :, :].repeat(batch_size, 1, 1).cuda() #.numpy()
            # # reshape to grid, and swap axes (permute x and z), B x reso x reso x reso x 3
            raster_pts = raster_pts.reshape(batch_size, self.reso_grid, self.reso_grid, self.reso_grid, 3)
            raster_pts = raster_pts.permute(0, 3, 2, 1, 4)

            # # reshape back to B x N x 3
            raster_pts = raster_pts.reshape(batch_size, -1, 3)

            ind = index['grid'][0].squeeze()
            ind0 = ind[0]
            # ind1 = ind[1]
            ind1 = ind[3]
            ind2 = ind[2]
            pt0 = raster_pts[0][ind0]
            pt1 = raster_pts[0][ind1]
            pt2 = raster_pts[0][ind2]
            sph0 = trimesh.creation.uv_sphere(0.005).apply_translation(pt0.detach().cpu().numpy().squeeze())
            sph1 = trimesh.creation.uv_sphere(0.005).apply_translation(pt1.detach().cpu().numpy().squeeze())
            sph2 = trimesh.creation.uv_sphere(0.005).apply_translation(pt2.detach().cpu().numpy().squeeze())
            util.meshcat_trimesh_show(self.mc_vis, 'scene/sph0', sph0, (255, 0, 0))
            util.meshcat_trimesh_show(self.mc_vis, 'scene/sph1', sph1, (0, 255, 0))
            util.meshcat_trimesh_show(self.mc_vis, 'scene/sph2', sph2, (0, 0, 255))

            ind_idx0 = torch.where(ind == ind0)[0]
            ind_idx1 = torch.where(ind == ind1)[0]
            ind_idx2 = torch.where(ind == ind2)[0]

            p0, pp0 = p[0].squeeze(), pp[0].squeeze()
            pind0 = p0[ind_idx0]
            pind1 = p0[ind_idx1]
            pind2 = p0[ind_idx2]

            ppind0 = pp0[ind_idx0]
            ppind1 = pp0[ind_idx1]
            ppind2 = pp0[ind_idx2]

            util.meshcat_pcd_show(self.mc_vis, p0.detach().cpu().numpy(), (0, 0, 0), 'scene/p0')

            util.meshcat_pcd_show(self.mc_vis, pind0.detach().cpu().numpy(), (255, 0, 0), 'scene/pind0')
            util.meshcat_pcd_show(self.mc_vis, pind1.detach().cpu().numpy(), (0, 255, 0), 'scene/pind1')
            util.meshcat_pcd_show(self.mc_vis, pind2.detach().cpu().numpy(), (0, 0, 255), 'scene/pind2')
            util.meshcat_pcd_show(self.mc_vis, ppind0.detach().cpu().numpy(), (255, 0, 0), 'scene/ppind0')
            util.meshcat_pcd_show(self.mc_vis, ppind1.detach().cpu().numpy(), (0, 255, 0), 'scene/ppind1')
            util.meshcat_pcd_show(self.mc_vis, ppind2.detach().cpu().numpy(), (0, 0, 255), 'scene/ppind2')

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
        if 'grid_sparse' in self.plane_type:
            fea['grid_sparse'] = self.generate_grid_features_sparse(index, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p_coords, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p_coords, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p_coords, c, plane='yz')
            
        if debug_viz:
            output_channel_dim = fea['grid'].shape[1]
            fea_grid = fea['grid'].permute(0, 2, 3, 4, 1)[0].reshape(-1, output_channel_dim)
            fea_norm = torch.norm(fea_grid, p=2, dim=-1)

            vals, inds = torch.topk(fea_norm, k=100)
            # inds = torch.where(fea_norm > 0.0)[0]
            sz_base = 1.1/32
            for i, idx in enumerate(inds):
                pt = raster_pts[0][idx].detach().cpu().numpy()
                box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
                util.meshcat_trimesh_show(self.mc_vis, f'scene/norm/box_{i}', box, opacity=0.3)

            # filled_raster_pts = raster_pts.permute(0, 2, 1).gather(dim=2, index=index['grid'].expand(-1, 3, -1)).permute(0, 2, 1)
            # filled_raster_pts = filled_raster_pts[0].detach().cpu().numpy().squeeze()
            # for i, pt in enumerate(filled_raster_pts):
            #     box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
            #     util.meshcat_trimesh_show(self.mc_vis, f'scene/filled/box_{i}', box, (255, 0, 0), opacity=0.3)

            print(f'Here with debug viz')
            from IPython import embed; embed()

        end = time.time()
        dur = end - start
        # print(f'[Scene Encoder] Time: {dur:.5f}')
            
        return fea

