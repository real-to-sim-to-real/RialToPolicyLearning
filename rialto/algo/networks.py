import numpy as np

import rlutil.torch as torch
import rlutil.torch.nn as nn
import torch.nn.functional as F
import rlutil.torch.pytorch_util as ptu
from rlkit.torch.distributions import ( TanhNormal,
)
from rlkit.torch.networks import Mlp
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
)
import torch.distributions
from rialto import policy

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

import rlutil.torch.pytorch_util as ptu



class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

# Define the forward model
class RewardModel(nn.Module):
    def __init__(self, env, obs_dim, fourier=False, normalize=False, layers=[600,600], is_complex_maze=False):
        super().__init__()
        self.is_complex_maze =is_complex_maze
        print("Is complex maze reward model", self.is_complex_maze)
        self.normalize = normalize
        
        if normalize and isinstance(env.wrapped_env, RavensGoalEnvPickOrPlace):
            assert env.wrapped_env.num_blocks == 1
            print("Is goal env pick or place")
            obs_low = np.array([0.25, -0.5, 0, 0.25, -0.5])
            obs_high = np.array([0.75, 0.5, 1, 0.75, 0.5])
            self.obs_space_mean = torch.tensor((obs_low+obs_high)/2, dtype=torch.float32)
            self.obs_space_range = torch.tensor((obs_high-obs_low)/2, dtype=torch.float32)

        if self.is_complex_maze:
            dim = obs_dim+2
        else:
            dim = 2*obs_dim

        self.trunk = FCNetwork(dim, 1,fourier, layers)#mlp(2*obs_dim, hidden_dim, 1, hidden_depth)
        self.outputs = dict()


    def forward(self, obs, goal):
        #print("obs shape", obs.shape)
        #print("goal shape", goal.shape)
        if self.is_complex_maze:
            goal = goal[:,:2]

        if self.normalize:
            obs = torch.tensor((obs- self.obs_space_mean)/self.obs_space_range, dtype=torch.float32)
            goal = torch.tensor((goal- self.obs_space_mean)/self.obs_space_range, dtype=torch.float32)

        state = torch.concat([obs, goal], axis=-1).to(ptu.CUDA_DEVICE)
        #print("State shape", state.shape)
        rpred = self.trunk(state)
        return rpred

# Define the forward model
class RewardModelHumanPreferences(nn.Module):
    def __init__(self, obs_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, 1, hidden_depth)
        self.outputs = dict()

    def forward(self, state):
        #print("obs shape", obs.shape)
        #print("goal shape", goal.shape)
        #print("State shape", state.shape)
        rpred = self.trunk(state)
        return rpred
    
class VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 2)
        self.conv2 = nn.Conv2d(16, 16, 5, stride = 2)
        self.conv3 = nn.Conv2d(16, 8, 5, stride = 2)

    def forward(self, obs):
        obs = (obs / 255.0) - 0.5

        obs = F.relu(self.conv1(obs))
        obs = F.relu(self.conv2(obs))
        obs = F.relu(self.conv3(obs))
        
        obs = torch.flatten(obs, 1)
        return obs

class RewardModelImages(nn.Module):
    """
    Neural Network such that, given a state image and a goal image, returns a reward which, ideally,
    correlates with the "distance" between the state and the goal. 
    """
    def __init__(self, image_size, fourier=False, layers=[600, 600]):
        super().__init__()
        print("INIT RW MODEL", fourier, layers)
        # Note: empirically it seems that sharing the encoder between state and image is beneficial. 
        self.encoder = VisualEncoder()
        dim = 2 * self.encoder(torch.zeros((1, 3, image_size, image_size))).shape[1]

        self.trunk = FCNetwork(dim, 1, fourier, layers)

    def forward(self, obs, goal):
        obs = self.encoder(obs.to(ptu.CUDA_DEVICE))
        goal = self.encoder(goal.to(ptu.CUDA_DEVICE))

        state = torch.cat((obs, goal), dim=1)
        
        rpred = self.trunk(state)
        return rpred


class CompareImages(nn.Module):
    """
    Neural Network such that, given two state images aims to return the "probability" that
    one can go from one image to the other in at most 2 actions (check ). 
    """
    def __init__(self, image_size, fourier=False, layers=[600,600]):
        super().__init__()
        # Note: empirically it seems that sharing the encoder between state and image is beneficial. 
        self.encoder = VisualEncoder()
        dim = 2 * self.encoder(torch.zeros((1, 3, image_size, image_size))).shape[1]

        self.trunk = FCNetwork(dim, 1, fourier, layers)

    def forward(self, obs, goal):
        obs = self.encoder(obs.to(ptu.CUDA_DEVICE))
        goal = self.encoder(goal.to(ptu.CUDA_DEVICE))

        state = torch.cat((obs, goal), dim=1)
        
        rpred = self.trunk(state)
        return torch.sigmoid(rpred)
    
class FCNetwork(nn.Module):
    """
    A fully-connected network module
    """
    def __init__(self, dim_input, dim_output, fourier=False, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork, self).__init__()
        net_layers = []
        self.fourier = fourier
        if fourier:
            dim = dim_input*40
        else:
            dim = dim_input
        self.outputs = dict()
        
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(0.4))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)
        self.apply(weight_init)
        if fourier:
            self.obs_f = LFF(dim_input, dim_input*40)


    def forward(self, states):
        if self.fourier:
            states = self.obs_f(states)

        return self.network(states)

class GaussianFCNetwork(Mlp, TorchStochasticPolicy):

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            fourier_size=40,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim*fourier_size,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.fourier_size = fourier_size
        self.obs_f = LFF(obs_dim, obs_dim*fourier_size)
        
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        obs = obs.to(ptu.CUDA_DEVICE)
        h = self.obs_f(obs)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(ptu.CUDA_DEVICE)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class CBCNetwork(nn.Module):
    """
    A fully connected network which appends conditioning to each hidden layer
    """
    def __init__(self, dim_input, dim_conditioning, dim_output, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0, add_conditioning=True):
        super(CBCNetwork, self).__init__()
        
        self.dropout = bool(dropout != 0)
        self.add_conditioning = add_conditioning

        net_layers = torch.nn.ModuleList([])
        dim = dim_input + dim_conditioning
        
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if self.dropout:
              net_layers.append(torch.nn.Dropout(dropout))
          if add_conditioning:
            dim = layer_size + dim_conditioning
          else:
            dim = layer_size

        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers

    def forward(self, states, conditioning):
        output = torch.cat((states, conditioning), dim=1)
        mod = 3 if self.dropout else 2
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i % mod == mod - 1 and self.add_conditioning:
                output = torch.cat((output, conditioning), dim=1)
        return output
        
class MultiInputNetwork(nn.Module):
    def __init__(self, input_shapes, dim_out, input_embeddings=None, layers=[512, 512], freeze_embeddings=False):
        super(MultiInputNetwork, self).__init__()
        if input_embeddings is None:
            input_embeddings = [Flatten() for _ in range(len(input_shapes))]

        self.input_embeddings = input_embeddings
        self.freeze_embeddings = freeze_embeddings    
        
        dim_ins = [
            embedding(torch.tensor(np.zeros((1,) + input_shape))).size(1)
            for embedding, input_shape in zip(input_embeddings, input_shapes)
        ]
        
        full_dim_in = sum(dim_ins)
        self.net = FCNetwork(full_dim_in, dim_out, layers=layers)
    
    def forward(self, *args):
        assert len(args) == len(self.input_embeddings)
        embeddings = [embed_fn(x) for embed_fn,x in zip(self.input_embeddings, args)]
        embed = torch.cat(embeddings, dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()
        return self.net(embed)


class StateGoalNetwork(nn.Module):
    def __init__(self, env, dim_out=1, state_embedding=None, continuous_action_space=False, fourier=False, goal_embedding=None, layers=[512, 512], max_horizon=None, freeze_embeddings=False, add_extra_conditioning=False, dropout=0, is_complex_maze= False):
        super(StateGoalNetwork, self).__init__()
        self.max_horizon = max_horizon
        self.continuous_action_space = continuous_action_space
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()
        
        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings
        self.is_complex_maze = is_complex_maze


        print("Is complex maze", self.is_complex_maze)

        state_dim_in = self.state_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]
        if self.is_complex_maze:
            goal_dim_in = 2
        else: 
            goal_dim_in = self.goal_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]

        dim_in = state_dim_in + goal_dim_in
        if continuous_action_space:
            self.net = GaussianFCNetwork(hidden_sizes=layers, obs_dim=dim_in, action_dim=env.action_space.shape[0])
        elif max_horizon is not None:
            print("network with horizon")
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, fourier=fourier, add_conditioning=add_extra_conditioning, dropout=dropout)
        else:
            self.net = FCNetwork(dim_in, dim_out, fourier=fourier, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
        if self.is_complex_maze:
            goal = goal[ :, :2]
        goal = self.goal_embedding(goal)
        embed = torch.cat((state, goal), dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()

        if self.max_horizon is not None:
            horizon = self.process_horizon(horizon)
            output = self.net(embed, horizon)
        else:
            output = self.net(embed)
        return output
    
    def process_horizon(self, horizon):
        # Todo add format options
        return horizon

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    one_hot_mask = (torch.arange(0, num_classes)
                               .to(ptu.CUDA_DEVICE)
                               .long()
                               .repeat(batch_size, 1)
                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)

def cross_entropy_with_weights(logits, target, weights=None, label_smoothing=0):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = torch.logsumexp(logits, dim=1) - (1-label_smoothing) * class_select(logits, target) - label_smoothing * logits.mean(dim=1)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean', label_smoothing=0):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.label_smoothing = label_smoothing

    def forward(self, input, target, weights=None):
        ce = cross_entropy_with_weights(input, target, weights, self.label_smoothing)
        if self.aggregate == 'sum':
            return ce.sum()
        elif self.aggregate == 'mean':
            return ce.mean()
        elif self.aggregate is None:
            return ce

class DiscreteStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, normalize=False, **kwargs):
        super(DiscreteStochasticGoalPolicy, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = StateGoalNetwork(env, dim_out=self.dim_out, **kwargs)        
        self.obs_space_mean = 0#(self.observation_space.high + self.observation_space.low)/2
        self.obs_space_range = 1#(self.observation_space.high - self.observation_space.low)/2
        self.normalize = normalize

    def forward(self, obs, goal, horizon=None):
        obs = obs.to(ptu.CUDA_DEVICE)
        goal = goal.to(ptu.CUDA_DEVICE)
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        if self.normalize:
            assert False
            print("Attention: Normalization not implemented")
            obs = torch.tensor((obs- self.obs_space_mean)/self.obs_space_range, dtype=torch.float32) 
            goal = torch.tensor((goal- self.obs_space_mean)/self.obs_space_range, dtype=torch.float32) 
        else:
            obs = torch.tensor(obs, dtype=torch.float32).to(ptu.CUDA_DEVICE)
            goal = torch.tensor(goal, dtype=torch.float32).to(ptu.CUDA_DEVICE)

        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)

        logits = self.forward(obs, goal, horizon)
     
        noisy_logits = logits #* (1 - noise)
        probs = torch.softmax(noisy_logits, 1)
        
        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()

        return ptu.to_numpy(samples)
    
    def nll(self, obs, goal, actions, horizon=None):        
        logits = self.forward(obs, goal, horizon=horizon)
        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, )
    
    def loss_regression(self, obs, goal, actions, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        return nn.MSELoss()(logits, actions)

    def probabilities(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(probs * logits, 1)

    def process_horizon(self, horizon):
        return horizon

    def get_actions_logprob_entropy(self, obs):
        logits = self.forward(obs, obs)
        # probs = torch.softmax(logits, 1)
        action_dist = torch.distributions.categorical.Categorical(logits=logits)
        actions = action_dist.sample()
        log_prob = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        actions = actions.reshape(-1,1)
        return actions, action_dist, log_prob, entropy

    def logprob_entropy(self, state, actions):
        _, action_dist, _, _ = self.get_actions_logprob_entropy(state)
        log_prob = action_dist.log_prob(actions.reshape(-1))
        entropy = action_dist.entropy()
        return actions, action_dist, log_prob, entropy

class ContinuousStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, normalize=False, **kwargs):
        super(ContinuousStochasticGoalPolicy, self).__init__()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.dim_out = env.action_space.shape
        self.net = StateGoalNetwork(env, continuous_action_space=True, dim_out=self.dim_out, **kwargs)  
        if isinstance(env.wrapped_env, RavensGoalEnvPickOrPlace):
            assert env.wrapped_env.num_blocks == 1 or not normalize
            print("Is goal env pick or place")
            obs_low = np.array([0.25, -0.5, 0, 0.25, -0.5])
            obs_high = np.array([0.75, 0.5, 1, 0.75, 0.5])
            action_low = obs_low[:2]
            action_high = obs_high[:2]
            self.obs_space_mean = (obs_low+obs_high)/2
            self.obs_space_range = (obs_high-obs_low)/2
            self.action_space_mean = (action_low + action_high)/2
            self.action_space_range = (action_high - action_low)/2
        else:
            self.obs_space_mean = (self.observation_space.high + self.observation_space.low)/2
            self.obs_space_range = (self.observation_space.high - self.observation_space.low)/2
            self.action_space_mean = (self.action_space.high + self.action_space.low)/2
            self.action_space_range = (self.action_space.high - self.action_space.low)/2
        self.normalize = normalize
        print("Is normalized", self.normalize)

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        
        if self.normalize:
            obs = (obs- self.obs_space_mean)/self.obs_space_range
            goal = (goal- self.obs_space_mean)/self.obs_space_range
            print("Normalized goal and obs", obs, goal)

        obs = torch.tensor(obs, dtype=torch.float32) 
        goal = torch.tensor(goal, dtype=torch.float32) 

        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        dist = self.forward(obs, goal, horizon)
        
        samples = dist.sample()

        samples = ptu.to_numpy(samples)
        if self.normalize:
            print("samples before normalized", samples)
            samples = samples*self.action_space_range+self.action_space_mean
        
        return samples 
    
    def nll(self, obs, goal, actions, horizon=None):        
        logits = self.forward(obs, goal, horizon=horizon)
        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, )
    
    def loss_regression(self, obs, goal, actions, horizon=None):
        dist = self.forward(obs, goal, horizon=horizon)   
        policy_logpp = dist.log_prob(actions, )
        logp_loss = -policy_logpp.mean()
        policy_loss = logp_loss
        return policy_loss

    def probabilities(self, obs, goal, horizon=None):
        probs = self.forward(obs, goal, horizon=horizon)
        #probs = self.net.net._get_dist_from_np(obs_np[None])

        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        #probs = self.net.net._get_dist_from_np(obs_np[None])

        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(logits, 1)

    def process_horizon(self, horizon):
        return horizon

class IndependentDiscretizedStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(IndependentDiscretizedStochasticGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        self.n_dims = self.action_space.n_dims
        self.granularity = self.action_space.granularity
        dim_out = self.n_dims * self.granularity
        self.net = StateGoalNetwork(env, dim_out=dim_out, **kwargs)        

    def flattened(self, tensor):
        tensor = tensor.to(ptu.CUDA_DEVICE)
        # tensor expected to be n x self.n_dims
        multipliers = self.granularity ** torch.tensor(np.arange(self.n_dims)).to(ptu.CUDA_DEVICE)
        flattened = (tensor * multipliers).sum(1)
        return flattened.int()
    
    def unflattened(self, tensor):
        # tensor expected to be n x 1
        digits = []
        output = tensor
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            output = output // self.granularity
        uf = torch.stack(digits, dim=-1)
        return uf

    def forward(self, obs, goal, horizon=None):
        obs = obs.to(ptu.CUDA_DEVICE)
        goal = goal.to(ptu.CUDA_DEVICE)
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0, marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        #noisy_logits = logits  * (1 - noise)
        noisy_logits = logits
        probs = torch.softmax(noisy_logits, 2)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        samples = self.flattened(samples)
        if greedy:
            samples = ptu.to_numpy(samples)
            random_samples = np.random.choice(self.action_space.n, size=len(samples))
            return np.where(np.random.rand(len(samples)) < noise,
                    random_samples,
                    samples,
            )
        return ptu.to_numpy(samples)
    
    def nll(self, obs, goal, actions, horizon=None):        
        actions_perdim = self.unflattened(actions)
        # print(actions, self.flattened(actions_perdim))
        actions_perdim = actions_perdim.view(-1)

        logits = self.forward(obs, goal, horizon=horizon)
        logits_perdim = logits.view(-1, self.granularity)
        
        loss_perdim = CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits_perdim, actions_perdim, weights=None)
        loss = loss_perdim.reshape(-1, self.n_dims)
        return loss.sum(1)
    
    def probabilities(self, obs, goal, horizon=None):
        """
        TODO(dibyaghosh): actually implement
        """
        raise NotImplementedError()

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        probs = torch.softmax(logits, 2)
        Z = torch.logsumexp(logits, dim=2)
        return (Z - torch.sum(probs * logits, 2)).sum(1)


#Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)



# Define the forward model
class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, fourier):
        super().__init__()
        input_dim = obs_dim*40 if fourier else obs_dim
        self.trunk = mlp(input_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(weight_init)
        self.fourier = fourier
        self.obs_f = LFF(obs_dim, obs_dim*40)

    def forward(self, obs):
        if self.fourier:
            obs = self.obs_f(obs)
        next_pred = self.trunk(obs)
        return next_pred


# Define the forward model for nonlinear hypernet
class TransformPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, action_dim, hidden_depth)
        self.outputs = dict()

    # Going forward with passed in parameters
    def forward_parameters(self, in_val, parameters=None):
        if parameters is None:
            parameters = list(self.parameters())

        output = in_val
        for params_idx in range(0, len(parameters) - 2, 2):
            w = parameters[params_idx]
            b = parameters[params_idx + 1]
            output = F.linear(output, w, b)
            output = F.relu(output)
        w = parameters[-2]
        b = parameters[-1]
        output = F.linear(output, w, b)
        return output



class LFF(nn.Linear):
    def __init__(self, inp, out, bscale=0.5):
        #out = 40*inp
        super().__init__(inp, out)
        nn.init.normal(self.weight, std=bscale/inp)
        nn.init.uniform(self.bias, -1.0, 1.0)

    def forward(self, x):
        x = np.pi * super().forward(x)
        return torch.sin(x)


