import numpy as np

import torch
import torch.nn as nn

import time

try:
    from torchsparse import nn as spnn
    from torchsparse import SparseTensor
    from torchsparse.backbones import SparseResNet21D
    from torchsparse.backbones.resnet import SparseResNet as SparseResNetBackbone
except ImportError as e:
    print(f'Could not import from torchsparse - sparse models unavailable ({e})')

from utils import coords_to_sparse
import wandb

class LayerNorm(nn.LayerNorm):
    def forward(self, input: SparseTensor) -> SparseTensor:
        coords, feats, stride = input.coords, input.feats, input.stride
        out_feats = super(LayerNorm, self).forward(feats)
        output = SparseTensor(coords=coords, feats=out_feats, stride=stride)
        return output

class SparseResNet(nn.Module):
    def __init__(self, in_channels, out_channels=None, mlp_spec=[128, 256], pool="avg"):
        super(SparseResNet, self).__init__()

        self.net = SparseResNet21D(in_channels=in_channels).to("cuda")
        if pool == "avg":
            self.pooling_layer = spnn.GlobalAvgPool()
        elif pool == "max":
            self.pooling_layer = spnn.GlobalMaxPool()
    def forward(self, inputs):
        all_outputs = self.net(inputs)
        x =  self.pooling_layer(all_outputs[-1])
        # except:
        #     x = torch.zeros((torch.max(inputs.coords[:,0]).item()+1, 128)).to("cuda")
        #     wandb.log({"Error in Encoder": 1})
        return x

class SparseResNetBigger(SparseResNetBackbone):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            blocks=[
                (3, 16, 3, 1),
                (3, 32, 3, 2),
                (3, 64, 3, 1),
                (3, 128, 3, 3),
                (3, 256, 3, 1),
                (3, 512, 3, 1),
                (1, 512, (1, 3, 1), (1, 2, 1)),
            ],
            **kwargs,
        )

class LargeSparseResNet(nn.Module):
    def __init__(self, in_channels, out_channels=None, mlp_spec=[128, 256], pool="avg"):
        super(LargeSparseResNet, self).__init__()

        self.net = SparseResNetBigger(in_channels=in_channels).to("cuda")
        self.max_pooling_layer = spnn.GlobalAvgPool()
        self.avg_pooling_layer = spnn.GlobalMaxPool()
        net_layers = []
        dim = 1024
        layers = [256, 128]
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(torch.nn.ReLU())
          dim = layer_size
        self.layers = net_layers
        self.mlp = torch.nn.Sequential(*net_layers).to("cuda")

    def forward(self, inputs):
        all_outputs = self.net(inputs)
        xmax =  self.max_pooling_layer(all_outputs[-1])
        xavg =  self.avg_pooling_layer(all_outputs[-1])
        x = torch.hstack([xmax, xavg])
        x = self.mlp(x)
        # except:
        #     x = torch.zeros((torch.max(inputs.coords[:,0]).item()+1, 128)).to("cuda")
        #     wandb.log({"Error in Encoder": 1})
        return x


class SparseCNN(nn.Module):
    def __init__(self, in_channels, out_channels=None, mlp_spec=[128, 256], pool="avg"):
        super(SparseCNN, self).__init__()

        modules = [
            spnn.Conv3d(in_channels, 128, kernel_size=4, stride=2),
            LayerNorm(128, eps=1e-6),
            spnn.ReLU(True),
            spnn.Conv3d(128, 128, kernel_size=3, stride=2),
            LayerNorm(128, eps=1e-6),
            spnn.ReLU(True),
        ]
        self.net = nn.Sequential(*modules)

        if pool == "avg":
            self.pooling_layer = spnn.GlobalAvgPool()
        elif pool == "max":
            self.pooling_layer = spnn.GlobalMaxPool()

    def forward(self, inputs):
        try:
            all_outputs = self.net(inputs)
            x =  self.pooling_layer(all_outputs[-1])
        except:
            x = torch.zeros((torch.max(inputs.coords[:,0]).item()+1, 128)).to("cuda")
            wandb.log({"Error in Encoder": 1})
        return x
        
class SparseConvPolicy(nn.Module):
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
            emb_size=128,
            dropout = 0,
            nonlinearity=torch.nn.ReLU,
            use_state=False,
            state_dim=9, # tool position, tool orientation, open 3 + 4 + 2
            encoder_type="resnet",
            pool="avg",
            device="cuda"
        ):    
        super(SparseConvPolicy, self).__init__()

        self.augment_obs = augment_obs
        self.augment_points = augment_points
        self.use_state = use_state
        self.pad_points = pad_points

        self.rgb_feats = rgb_feats
        self.emb_size  = emb_size
        mlp_spec=np.concatenate([emb_layers, [self.emb_size]])
        self.act_size = act_size

        if encoder_type == "resnet":
            self.point_encoder = SparseResNet(in_channels=in_channels, out_channels=self.emb_size, mlp_spec=mlp_spec, pool=pool).to("cuda")
        elif encoder_type == "large_resnet":
            self.point_encoder = LargeSparseResNet(in_channels=in_channels, out_channels=self.emb_size, mlp_spec=mlp_spec, pool=pool).to("cuda")
        elif encoder_type == "cnn":
            self.point_encoder = SparseCNN(in_channels=in_channels, out_channels=self.emb_size, mlp_spec=mlp_spec, pool=pool).to("cuda")

        net_layers = []
        dim = self.emb_size
        if self.use_state:
            dim += state_dim
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(dropout))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, act_size))
        self.layers = net_layers
        self.mlp = torch.nn.Sequential(*net_layers).to("cuda")

    def forward(self, x):
        # Input shape np.array(T, B, N, 3) torch.tensor(T, B, dim)
        (coords, feats, obs) = x
        
        # Input shape ( B, N, 3)
        B, N, D_c = coords.shape
        D_f = feats.shape[-1]

        try:

            sparse_tensor = coords_to_sparse(coords, feats, unpad=self.pad_points, apply_augment=self.augment_points, rgb_feats=self.rgb_feats)
            # print("inside forward")
            # import IPython
            # IPython.embed()
            # inputs = SparseTensor(coords=torch.tensor(coords, dtype=torch.int), feats = torch.tensor(feats, dtype=torch.float)).to("cuda")
            # print("hi",sparse_tensor.coords.shape)
            # eval_data = np.load("eval_pcd.npy")

            # import open3d as o3d
            # i=0
            # for i in range(200):
            #     pcd = o3d.geometry.PointCloud()
            #     print("num_points", torch.where(sparse_tensor.coords[:,0]==i)[0].shape[0])
            #     pcd.points = o3d.utility.Vector3dVector(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i)[0],1:].cpu().numpy())
            #     pcd_eval = o3d.geometry.PointCloud()
            #     print("num_points", torch.where(sparse_tensor.coords[:,0]==i)[0].shape[0])
            #     pcd_eval.points = o3d.utility.Vector3dVector(eval_data[np.where(eval_data[:,0]==i)[0],1:])
                
            #     o3d.visualization.draw_geometries([pcd, pcd_eval])

            # import open3d as o3d
            # pcds = []
            # for i in range(10):
            #     idx = 24
            #     pcd = o3d.geometry.PointCloud()
            #     new_data = unpad_points(data[idx])
            #     pcd.points = o3d.utility.Vector3dVector(new_data)
            #     pcds.append(pcd)
            # o3d.visualization.draw_geometries(pcds)

            # # import open3d as o3d
            # i=1
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i)[0],1:].cpu().numpy())
            # pcd2 = o3d.geometry.PointCloud()
            # pcd2.points = o3d.utility.Vector3dVector(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i+1)[0],1:].cpu().numpy())
            # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i+1)[0],1:].cpu().numpy()))
            # o3d.visualization.draw_geometries([pcd])

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sparse_tensor.coords[:,1:].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])

            # np.save(f"pcd_online_distill.npy", sparse_tensor.coords[:,1:].cpu().numpy())
            # print("PCD range", torch.min(sparse_tensor.coords, axis=0).values, torch.max(sparse_tensor.coords, axis=0).values)
            # torch.where(points_embed[-1].coords[:,0]==5)[0].shape

            points_embed = self.point_encoder(sparse_tensor)
            points_embed = points_embed.reshape(B, -1)

            if self.use_state:
                mlp_input = torch.hstack([points_embed, obs])
            else:
                mlp_input = points_embed

            logits = self.mlp(mlp_input)

        except:
            import IPython
            IPython.embed()

        return logits
    
    def compute_loss(self, coords, feats, obs, actions):

        logits = self.forward((coords, feats, obs))
        loss = nn.CrossEntropyLoss( reduction='mean')(logits, actions)
        return loss

class SparseRNNConvPolicy(nn.Module):
    def __init__(
            self,
            in_channels,
            # obs_size,
            act_size,
            hidden_size=256,
            emb_size=128,
            augment_obs=False,
            augment_points=False,
            rgb_feats=False,
            pad_points=False,
            layers=[256,256],
            emb_layers=[64],
            dropout = 0,
            gru=False,
            nonlinearity=torch.nn.ReLU,
            use_state=False,
            state_dim=9,
            device="cuda",
            encoder_type="resnet"
        ):    
        super(SparseRNNConvPolicy, self).__init__()

        self.augment_obs = augment_obs
        self.augment_points = augment_points
        self.use_state = use_state
        self.pad_points = pad_points

        self.rgb_feats = rgb_feats
        self.hidden_size = hidden_size
        self.emb_size  = emb_size
        mlp_spec=np.concatenate([emb_layers, [self.emb_size]])
        self.act_size = act_size

        if encoder_type == "resnet":
            self.point_encoder = SparseResNet(in_channels=in_channels, out_channels=self.emb_size, mlp_spec=mlp_spec).to("cuda")
        elif encoder_type == "cnn":
            self.point_encoder = SparseCNN(in_channels=in_channels, out_channels=self.emb_size, mlp_spec=mlp_spec).to("cuda")

        if gru:
            self.rnn = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0, bidirectional=False, device=None, dtype=None).to("cuda")
        else:
            self.rnn = nn.RNN(input_size=emb_size, hidden_size=hidden_size, num_layers=1, nonlinearity='relu', bias=True, batch_first=True, dropout=0.0, bidirectional=False, device=None, dtype=None).to("cuda")
        net_layers = []
        dim = self.hidden_size
        if self.use_state:
            dim += state_dim

        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(dropout))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, act_size))
        self.layers = net_layers
        self.mlp = torch.nn.Sequential(*net_layers).to("cuda")


    def init_belief(self, batch_size, device):
        self.belief = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        return self.belief

    def forward(self, x, init_belief=True):
        # Input shape np.array(T, B, N, 3) torch.tensor(T, B, dim)
        (coords, feats, obs) = x
        
        # Input shape ( B, N, 3)
        B, T, N, D_c = coords.shape
        D_f = feats.shape[-1]

        if init_belief:
            belief = self.init_belief(B, device=obs.device)
        else:
            belief = self.belief

        coords, feats = coords.reshape(T*B, N, D_c), feats.reshape(T*B, N, D_f)

        print("coords shape", coords.shape)
        sparse_tensor = coords_to_sparse(coords, feats, unpad=self.pad_points, apply_augment=self.augment_points, rgb_feats=self.rgb_feats)
        # inputs = SparseTensor(coords=torch.tensor(coords, dtype=torch.int), feats = torch.tensor(feats, dtype=torch.float)).to("cuda")
        # print("hi",sparse_tensor.coords.shape)
        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # i=0
        # for i in range(200):
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i)[0],1:].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])
        # i=1
        # np.save(f"pcd_online_distill.npy", sparse_tensor.coords[:,1:].cpu().numpy())
        # print("PCD range", torch.min(sparse_tensor.coords, axis=0).values, torch.max(sparse_tensor.coords, axis=0).values)
        # torch.where(points_embed[-1].coords[:,0]==5)[0].shape
        points_embed = self.point_encoder(sparse_tensor)
        points_embed = points_embed.reshape(B, T, -1)

        # Pass RNN
        beliefs, belief = self.rnn(points_embed, belief)
        # for t in range(T):
        #     belief = self.rnn(points_embed[t], belief)
        #     beliefs[t] = belief
            
        # beliefs = torch.stack(beliefs, dim=0)

        if self.use_state:
            mlp_input = torch.hstack([beliefs, obs])
        else:
            mlp_input = beliefs

        logits = self.mlp(mlp_input)#.chunk(2, -1)

        self.belief = belief.detach()

        return logits


    def predict_init_pose(self, x):
        # Input shape np.array(T, B, N, 3) torch.tensor(T, B, dim)
        (coords, feats, obs) = x
        
        # Input shape ( B, N, 3)
        B, N, D_c = coords.shape
        D_f = feats.shape[-1]

        # coords, feats = coords.reshape(B, N, D_c), feats.reshape(B, N, D_f)

        print("coords shape", coords.shape)
        sparse_tensor = coords_to_sparse(coords, feats, unpad=self.pad_points, apply_augment=self.augment_points, rgb_feats=self.rgb_feats)
        # inputs = SparseTensor(coords=torch.tensor(coords, dtype=torch.int), feats = torch.tensor(feats, dtype=torch.float)).to("cuda")
        # print("hi",sparse_tensor.coords.shape)
        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # i=0
        # for i in range(200):
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(sparse_tensor.coords[torch.where(sparse_tensor.coords[:,0]==i)[0],1:].cpu().numpy())
            # o3d.visualization.draw_geometries([pcd])
        # i=1
        # np.save(f"pcd_online_distill.npy", sparse_tensor.coords[:,1:].cpu().numpy())
        # print("PCD range", torch.min(sparse_tensor.coords, axis=0).values, torch.max(sparse_tensor.coords, axis=0).values)
        # torch.where(points_embed[-1].coords[:,0]==5)[0].shape
        points_embed = self.point_encoder(sparse_tensor)
        points_embed = points_embed.reshape(B, -1)

        # Pass RNN
        # beliefs, belief = self.rnn(points_embed, belief)
        # for t in range(T):
        #     belief = self.rnn(points_embed[t], belief)
        #     beliefs[t] = belief
            
        # beliefs = torch.stack(beliefs, dim=0)

        if self.use_state:
            mlp_input = torch.hstack([beliefs, obs])
        else:
            mlp_input = beliefs

        actions_mean, actions_std = self.mlp_pose(mlp_input)#.chunk(2, -1)

        return actions_mean, actions_std
    
    def compute_loss(self, coords, feats, obs, actions):
        logits = self.forward((coords, feats, obs))
        B, T, D = logits.shape
        logits = logits.reshape(B*T, D)
        actions = actions.reshape(B*T)
        loss = nn.CrossEntropyLoss( reduction='mean')(logits, actions)
        return loss

    def compute_loss_pose(self, coords, feats, obs, actions):
        actions_mean, actions_std = self.predict_init_pose((coords, feats, obs))
        # B, T, D = logits.shape
        # logits = logits.reshape(B*T, D)
        # actions = actions.)
        # TODO: check the same loss function as Marius was using
        loss = nn.CrossEntropyLoss( reduction='mean')(logits, actions)
        return loss
