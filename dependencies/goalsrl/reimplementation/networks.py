import numpy as np
import gym

import rlutil.torch as torch
import rlutil.torch.distributions
import rlutil.torch.nn as nn
import torch.nn.functional as F
import rlutil.torch.pytorch_util as ptu
from torch.nn.parameter import Parameter

from goalsrl import policy

device = ptu.default_device()
sqnorm = lambda x: torch.sum(x ** 2, 1)

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class RandomProjection(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(RandomProjection, self).__init__()
        self.layer = torch.nn.Linear(dim_input, dim_output, bias=False)
    def forward(self, states):
        embedding = torch.sin(self.layer(states))
        return embedding.detach()

class FCNetwork(nn.Module):
    """
    A fully-connected network module
    """
    def __init__(self, dim_input, dim_output, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork, self).__init__()
        net_layers = []
        dim = dim_input
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(0.4))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        return self.network(states)

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

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
                np.linspace(-1., 1., self.height),
                np.linspace(-1., 1., self.width)
                )
        pos_x = torch.tensor(pos_x.reshape(self.height*self.width), dtype=torch.float32)
        pos_y = torch.tensor(pos_y.reshape(self.height*self.width), dtype=torch.float32)
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)    

class CNNHead(nn.Module):
    """
    This is our default CNN architecture
    """        
    def __init__(self,
            image_size=32,
            spatial_softmax=False,
            output_size=32, # Note: This parameter is ignored if spatial_softmax=True
        ):
        
        super(CNNHead, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )



        self.do_spatial_softmax = spatial_softmax
        if spatial_softmax:
            self.spatial_softmax_layer = SpatialSoftmax(image_size // 4, image_size // 4, 32)
            # self.fc1 = nn.Linear(32 * 2, output_size)

        else:    
            self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        if self.do_spatial_softmax:
            out = self.spatial_softmax_layer(out)
        else:
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
        
        return out

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
    def __init__(self, env, dim_out=1, state_embedding=None, goal_embedding=None, layers=[512, 512], max_horizon=None, freeze_embeddings=False, add_extra_conditioning=False, dropout=0):
        super(StateGoalNetwork, self).__init__()
        self.max_horizon = max_horizon
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()
        
        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings

        state_dim_in = self.state_embedding(torch.tensor(env.observation_space.sample())[None]).size()[1]
        goal_dim_in = self.goal_embedding(torch.tensor(env.goal_space.sample())[None]).size()[1]

        dim_in = state_dim_in + goal_dim_in

        if max_horizon is not None:
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, add_conditioning=add_extra_conditioning, dropout=dropout)
        else:
            self.net = FCNetwork(dim_in, dim_out, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
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
    def __init__(self, env, **kwargs):
        super(DiscreteStochasticGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = StateGoalNetwork(env, dim_out=self.dim_out, **kwargs)        

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        logits = self.forward(obs, goal, horizon=horizon)
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 1)
        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)
        
    def nll(self, obs, goal, actions, horizon=None):        
        logits = self.forward(obs, goal, horizon=horizon)
        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, )
    
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

# Policies

class IndependentDiscretizedStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(IndependentDiscretizedStochasticGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        self.n_dims = self.action_space.n_dims
        self.granularity = self.action_space.granularity
        dim_out = self.n_dims * self.granularity
        self.net = StateGoalNetwork(env, dim_out=dim_out, **kwargs)        

    def flattened(self, tensor):
        """
        TODO(dibyaghosh): test implementation
        """
        # tensor expected to be n x self.n_dims
        multipliers = self.granularity ** torch.tensor(np.arange(self.n_dims))
        flattened = (tensor * multipliers).sum(1)
        return flattened.int()
    
    def unflattened(self, tensor):
        """
        TODO(dibyaghosh): test implementation
        """
        # tensor expected to be n x 1
        digits = []
        output = tensor
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            output = output // self.granularity
        uf = torch.stack(digits, dim=-1)
        return uf

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 2)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        samples = self.flattened(samples)
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

        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        probs = torch.softmax(noisy_logits, 2)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        probs = torch.softmax(noisy_logits, 2)
        Z = torch.logsumexp(logits, dim=2)
        return (Z - torch.sum(probs * logits, 2)).sum(1)

class ContinuousGaussianGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self,
            env,
            **kwargs
        ):

        super(ContinuousGaussianGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        action_dim = env.action_space.shape[0]
        self.max_action = self.action_space.high

        self.mean_net = StateGoalNetwork(env, dim_out=action_dim, **kwargs)        
        self.logstd_net = StateGoalNetwork(env, dim_out=action_dim, **kwargs)        

    def forward(self, obs, goal, horizon=None):
        return self.mean_net.forward(obs, goal, horizon=horizon), self.logstd_net(obs*0, goal*0, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        means, log_stds = self.forward(obs, goal, horizon=horizon)
        
        if greedy:
            samples = means
        else:
            stds = torch.exp(log_stds + noise) 
            samples = torch.distributions.normal.Normal(loc=means, scale=stds).sample()

        samples = np.clip(ptu.to_numpy(samples), -1 * self.max_action, self.max_action)
        return samples
    
    def nll(self, obs, goal, actions, horizon=None):        
        means, log_stds = self.forward(obs, goal, horizon=horizon)
        variances = torch.exp(2 * log_stds)
        nlls = log_stds + 0.5 * (means - actions)**2 / variances
        return nlls.sum(1)
        
    def process_horizon(self, horizon):
        return horizon

class ContinuousMixtureGaussianGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    """
        TODO: not actually tested yet (may contain bugs)
    """
    def __init__(self, env, n_components=4, **kwargs):
        super(ContinuousMixtureGaussianGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        
        self.action_dim = env.action_space.shape[0]
        self.max_action = self.action_space.high
        
        self.n_components = n_components

        self.mean_net = StateGoalNetwork(env, dim_out=self.action_dim * n_components, **kwargs)        
        self.logstd_net = StateGoalNetwork(env, dim_out=self.action_dim * n_components, **kwargs)        

        self.pi_net = StateGoalNetwork(env, dim_out=n_components, **kwargs)
    
    def forward(self, obs, goal, horizon=None):
        means, logstds = self.mean_net.forward(obs, goal, horizon=horizon), self.logstd_net(obs, goal, horizon=horizon)
        pis = self.pi_net(obs, goal, horizon=horizon)
        means = means.view(means.size(0), self.n_components, self.action_dim)
        logstds = logstds.view(logstds.size(0), self.n_components, self.action_dim)

        return means, logstds, pis

    def act_vectorized(self, obs, goal, horizon=None, greedy=False):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        means, log_stds, pis = self.forward(obs, goal, horizon=horizon)
        pi_probs = torch.softmax(pis, 1)
        if greedy:
            pi_indices = torch.argmax(pis, 1)
            means = means[pi_samples]
            samples = means
        else:
            pi_indices = torch.distributions.categorical.Categorical(probs=pi_probs).sample()
            stds = torch.exp(log_stds)
            means, stds = means[pi_samples], stds[pi_samples]
            samples = torch.distributions.normal.Normal(loc=means, scale=stds).sample()

        samples = np.clip(ptu.to_numpy(samples), -1 * self.max_action, self.max_action)
        return samples
    
    def nll(self, obs, goal, actions, horizon=None):        
        means, log_stds, pis = self.forward(obs, goal, horizon=horizon)
        pi_probs = torch.softmax(pis, 1)
        variances = torch.exp(2 * log_stds)
        nlls = log_stds + 0.5 * (means - actions)**2 / variances
        nlls = nlls.sum(2)
        nlls = torch.sum(nlls * pi_probs, 1)
        return nlls
    
    def process_horizon(self, horizon):
        return horizon

class EnsembleOfPolicies(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, policies, discrete=False):
        super(EnsembleOfPolicies, self).__init__()
        self.policies = torch.nn.ModuleList(policies)
        self.discrete = discrete
        assert self.discrete, "Not implemented otherwise"

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)

        all_probs = [policy.probabilities(obs, goal, horizon=horizon) for policy in self.policies]
        probabilities = torch.stack(all_probs, 1)
        probs = probabilities.mean(1)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        
        return ptu.to_numpy(samples)

    def nll(self, obs, goal, actions, horizon=None, mask=None):
        all_nlls = []
        for i, policy in enumerate(self.policies):
            all_nlls.append(policy.nll(obs, goal, actions, horizon=horizon))
        all_nlls = torch.stack(all_nlls, 1)
        if mask is not None:
            all_nlls = all_nlls * mask
        return all_nlls.mean(1)
    
    def disagreement(self, obs, goal, horizon=None):
        def quick_entropy(arr):
            return torch.sum(-1 * arr * torch.log(arr), -1)

        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        all_probs = [policy.probabilities(obs, goal, horizon=horizon) for policy in self.policies]
        probabilities = torch.stack(all_probs, 1)
        conditional_entropy = quick_entropy(probabilities).mean(1)
        entropy = quick_entropy(probabilities.mean(1))
        return entropy - conditional_entropy

class ContinuousGaussianFixedVarianceGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    """
    TODO(dibyaghosh): Refactor and put into ContinuousGaussianGoalPolicy
    """
    def __init__(self, env, logstd=0, **kwargs):
        super(ContinuousGaussianFixedVarianceGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        action_dim = env.action_space.shape[0]
        self.max_action = self.action_space.high

        self.mean_net = StateGoalNetwork(env, dim_out=action_dim, **kwargs)        
        self.logstd = logstd 

    def forward(self, obs, goal, horizon=None):
        return self.mean_net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        means = self.forward(obs, goal, horizon=horizon)
        log_stds = torch.zeros_like(means) + self.logstd
        
        if greedy:
            samples = means
        else:
            stds = torch.exp(log_stds) + noise
            samples = torch.distributions.normal.Normal(loc=means, scale=stds).sample()

        samples = np.clip(ptu.to_numpy(samples), -1 * self.max_action, self.max_action)
        return samples
    
    def nll(self, obs, goal, actions, horizon=None):        
        means = self.forward(obs, goal, horizon=horizon)
        log_stds = torch.zeros_like(means) + self.logstd
        variances = torch.exp(2 * log_stds)
        nlls = log_stds + 0.5 * (means - actions)**2 / variances
        return nlls.sum(1)
        
    def process_horizon(self, horizon):
        return horizon

# VAEs and other stuff

class VAE(nn.Module):
    def __init__(self, mean_network, logvar_network, decoder_network, is_image=False):
        super().__init__()
        self.mu = mean_network
        self.logvar = logvar_network
        self.decoder = decoder_network
        self.is_image = is_image
    
    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = self.z(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size()).to(device)
        return eps.mul(std).add_(mean)

    def loss(self, x, beta=1):
        x_reconstructed, mean, logvar = self.forward(x)
        recon_loss, kldiv_loss = self.reconstruction_loss(x_reconstructed, x), self.kl_divergence_loss(mean, logvar)
        loss = recon_loss + beta * kldiv_loss
        return loss, recon_loss, kldiv_loss

    def reconstruction_loss(self, x_reconstructed, x):
        if not self.is_image:
            return nn.MSELoss(size_average=False)(x_reconstructed, x) / x.size(0)
        else:
            return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).sum() / mean.size(0)
    
    def get_latents(self, x):
        return self.mu(x)
    
    def marginal_prob(self, x, n=10, beta=1):
        x_repeat = x.repeat(n, 1)
        mean, logvar = self.mu(x_repeat), self.logvar(x_repeat)
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(std.size()).to(device)
        sampled_latents = eps.mul(std).add_(mean)
        reconstructed_xs = self.decoder(sampled_latents)
        log_p_model = - 1/2 * sqnorm(eps) - logvar.sum(1) / 2
        log_p_global = -1/2 * sqnorm(sampled_latents)
        log_p_reconstructor = -1/2 * sqnorm(reconstructed_xs - x_repeat) / beta
        log_p_estimate = log_p_global + log_p_reconstructor - log_p_model
        log_p_estimate = log_p_estimate.view(n, -1)
        p_estimate = torch.exp(log_p_estimate).mean(0)
        return p_estimate

class UnFlatten(nn.Module):
    def forward(self, input,):
        return input.view(input.size(0), input.size(1), 1, 1)

class shittyDecoder(nn.Module):
    def __init__(self, image_size = 32, filter_num = 16, latent_size = 16):
        super(shittyDecoder, self).__init__()
        self.image_size = image_size
        self.net = FCNetwork(latent_size, image_size * image_size * 3, layers=[16, 128],)
    
    def forward(self, x):
        x = self.net(x)
        x = F.sigmoid(x.view(-1, 3, self.image_size, self.image_size))
        return x

class CNNVAEDecoder(nn.Module):
    def __init__(self, image_size = 32, filter_num = 16, latent_size = 16):
        super(CNNVAEDecoder, self).__init__()
        self.channel_size = 3
        self.output_h = image_size
        self.output_w = image_size
        self.filter_num = filter_num
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 16)
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(1, self.filter_num * 4, 4, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.filter_num * 4, self.filter_num * 2, 4, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.filter_num * 2, self.filter_num, 4, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.filter_num, 1, 4, 2),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 1, 4, 4)
        x = self.convt(x)
        x = x.view(-1, self.channel_size, self.output_h, self.output_w)
        return x

class CNNVAEEncoder(nn.Module):
    def __init__(self,
        image_size=32,
        spatial_softmax=False,
        latent_size=32,
    ):
        super(CNNVAEEncoder, self).__init__()
        self.cnn_head = CNNHead(image_size, spatial_softmax, output_size=64)
        mean_fc = nn.Linear(64, latent_size)
        logstd_fc = nn.Linear(64, latent_size)

        self.mean_encoder = nn.Sequential(self.cnn_head, mean_fc)
        self.logstd_encoder = nn.Sequential(self.cnn_head, logstd_fc)

    def forward(self,x):
        return self.mean_encoder(x)
    
    def mean(self):
        return self.mean_encoder
    
    def logstd(self):
        return self.logstd_encoder
