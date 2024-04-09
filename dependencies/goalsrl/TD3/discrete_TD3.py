import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from goalsrl.TD3 import utils
from goalsrl.reimplementation.networks import Flatten, MultiInputNetwork, CNNHead

import rlutil.torch.pytorch_util as ptu

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = ptu.default_device()

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

def masked_select(qvals, indices):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = qvals.size()
    one_hot_mask = (torch.arange(0, num_classes)
                               .long().to(device)
                               .repeat(batch_size, 1)
                               .eq(indices.data.repeat(num_classes, 1).t()))
    return qvals.masked_select(one_hot_mask).reshape(-1,1)


class Actor(nn.Module):
    def __init__(self, critic):
        super(Actor, self).__init__()
        self.critic = critic

    def forward(self, x, g):
        qvals = self.critic.Q1(x, g)
        probs = torch.softmax(qvals, 1)
        return probs

class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, n_actions, state_embedding=None, goal_embedding=None, detach_embeddings=False):
        super(Critic, self).__init__()

        if state_embedding is None:
            state_embedding = Flatten()
        self.state_embedding = state_embedding
        if goal_embedding is None:
            goal_embedding = Flatten()
        self.goal_embedding = goal_embedding
        self.detach_embeddings = detach_embeddings

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + goal_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, n_actions)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + goal_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, n_actions)
    
    def preprocess(self, x, g):
        x = self.state_embedding(x)
        g = self.goal_embedding(g)
        if self.detach_embeddings:
            x, g = x.detach(), g.detach()
        return x, g

    def forward(self, x, g, u=None):
        x, g = self.preprocess(x, g)
        xu = torch.cat([x, g], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        
        if u is not None:
            x1, x2 = masked_select(x1, u), masked_select(x2, u)
        return x1, x2


    def Q1(self, x, g, u=None):
        x, g = self.preprocess(x, g)
        xu = torch.cat([x, g], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        if u is not None:
            x1 = masked_select(x1, u)
        return x1 

class DQN(object):
    def __init__(self, env, lr=1e-3):
        state_embedding_fn = lambda: None
        goal_embedding_fn = lambda: None
        
        if len(env.observation_space.shape) > 1: # Images
            state_dim = 64
            goal_dim = 64
            imsize = env.observation_space.shape[1]
            state_embedding_fn = lambda: CNNHead(imsize, spatial_softmax=True, output_size=64)
            goal_embedding_fn = lambda: CNNHead(imsize, spatial_softmax=True, output_size=64)
        else:
            state_dim = np.prod(env.observation_space.shape)
            goal_dim = np.prod(env.goal_space.shape)
            
        n_actions = env.action_space.n
        
        self.critic = Critic(state_dim, goal_dim, n_actions, state_embedding_fn(), goal_embedding_fn()).to(device)
        self.critic_target = Critic(state_dim, goal_dim, n_actions, state_embedding_fn(), goal_embedding_fn()).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.actor = Actor(self.critic).to(device)
        self.actor_target = Actor(self.critic_target).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)


    def select_action(self, state, goal):
        state = torch.FloatTensor(state).to(device)[None]
        goal = torch.FloatTensor(goal).to(device)[None]
        probabilities = self.actor(state, goal).cpu().data.numpy().flatten()
        return probabilities.argmax()

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        import tqdm
        running_loss = None
        with tqdm.trange(iterations, leave=False) as ranger:
            for it in ranger:

                # Sample replay buffer 
                x, y, u, g, r, d = replay_buffer.sample(batch_size)
                state = torch.FloatTensor(x).to(device)
                action = torch.LongTensor(u).to(device)
                next_state = torch.FloatTensor(y).to(device)
                goal = torch.FloatTensor(g).to(device)
                done = torch.FloatTensor(1 - d).to(device)
                reward = torch.FloatTensor(r).to(device)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, goal)
                target_Q1 = torch.max(target_Q1, 1, keepdim=True)[0]
                target_Q2 = torch.max(target_Q2, 1, keepdim=True)[0]

                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward.view(-1,1) + (done * discount * target_Q).detach()
                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(state, goal, action)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                if running_loss is None:
                    running_loss = ptu.to_numpy(critic_loss)
                else:
                    running_loss = 0.9 * running_loss + 0.1 * ptu.to_numpy(critic_loss)

                if it % policy_freq == 0:
                    
                    # Compute actor loss
                    ranger.set_description('Critic loss: %f'% running_loss)

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))


    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
