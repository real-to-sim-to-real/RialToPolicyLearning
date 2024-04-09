import abc
from turtle import distance
from dependencies.multiworld.envs.env_util import get_stat_in_paths

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

from rialto.algo.networks import RewardModelHumanPreferences
from rialto.algo.buffer import ReplayBuffer, RewardModelBuffer
import numpy as np
import torch
import time
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from rialto.envs.room_env import PointmassGoalEnv



class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            reward_model_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            train_rewardmodel_freq=500,
            reward_model_num_samples = 100,
            display_plots=False,
            use_oracle=False,
            env_name="",
            start_epoch=0, # negative epochs are offline, positive epochs are online
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.use_oracle = use_oracle
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = start_epoch
        self.exploration_env = exploration_env
        self.reward_model_buffer = reward_model_buffer
        self.reward_model_batch_size=256
        self.timesteps = 0
        self.display_plots = display_plots
        self.env_name=env_name




        self.train_rewardmodel_freq = train_rewardmodel_freq
        print("observation shape", exploration_env.observation_space.shape)
        self.goal = exploration_env.goal
        evaluation_env.goal = self.goal
        self.reward_model = RewardModelHumanPreferences(exploration_env.observation_space.shape[0], 64, 2)
        if self.use_oracle:
            self.reward_model = self.oracle_model
            print("using oracle")
        else:
            self.reward_optimizer = torch.optim.Adam(list(self.reward_model.parameters()))

        self.reward_model_num_samples = reward_model_num_samples

    def get_avg_stats(self, paths):
        rewards = []
        distances = []
        for path in paths:
            distance = self.exploration_env.compute_shaped_distance(path['observations'][-1], self.goal)
            reward = int(distance < 0.05)

            rewards.append(reward)
            distances.append(distance)
        
        return rewards, distances

    def train_rewardmodel(self, num_epochs=400, batch_size=32, device='cuda' ):
         # Train standard goal conditioned policy

        loss_fn = torch.nn.CrossEntropyLoss() 
        losses_eval = []
        if self.use_oracle:
            self.reward_model.train()
        running_loss = 0.0
        
        # Train the model with regular SGD
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            start = time.time()
            
            achieved_states_1, achieved_states_2, goals ,labels = self.reward_model_buffer.sample_batch(batch_size)
            
            self.reward_optimizer.zero_grad()

            t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
            
            state1 = torch.Tensor(achieved_states_1[t_idx]).to(device)
            state2 = torch.Tensor(achieved_states_2[t_idx]).to(device)
            goal = torch.Tensor(goals[t_idx]).to(device)
            label_t = torch.Tensor(labels[t_idx]).long().to(device)

            g1g2 = torch.cat([self.reward_model(state1), self.reward_model(state2)], axis=-1)
            loss = loss_fn(g1g2, label_t)
            loss.backward()
            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())

        return running_loss, 0

    def train(self):
        """Negative epochs are offline, positive epochs are online"""
        for self.epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.offline_rl = self.epoch < 0
            self._begin_epoch(self.epoch)
            self._train()
            self._end_epoch(self.epoch)

    def oracle(self, state_1, state_2, goal):
        d1_dist = self.exploration_env.compute_shaped_distance(state_1, goal)#self.env.shaped_distance(state1, goal) # np.linalg.norm(state1 - goal, axis=-1)
        d2_dist = self.exploration_env.compute_shaped_distance(state_2, goal)#self.env.shaped_distance(state2, goal) # np.linalg.norm(state2 - goal, axis=-1)

        if d1_dist < d2_dist:
            return 0
        else:
            return 1

    def generate_pref_labels(self,):
        data1 = self.replay_buffer.random_batch(self.reward_model_num_samples) # TODO: add
        data2 = self.replay_buffer.random_batch(self.reward_model_num_samples) # TODO: add

        observations_1 = data1['observations']
        observations_2 = data2['observations']
        goals = []
        labels = []
        achieved_state_1 = []
        achieved_state_2 = []

        for state_1, state_2 in zip(observations_1, observations_2):
            labels.append(self.oracle(state_1, state_2, self.goal)) # oracle TODO: we will use human labels here

            achieved_state_1.append(state_1)
            achieved_state_2.append(state_2) 
            goals.append(self.goal)

        achieved_state_1 = np.array(achieved_state_1)
        achieved_state_2 = np.array(achieved_state_2)
        goals = np.array(goals)
        labels = np.array(labels)
    
        return achieved_state_1, achieved_state_2, goals, labels
    
    def collect_and_train_rewardmodel(self, device='cuda'):
        print("Collecting and training rewardmodel")
        # TODO: we are gonna substitute generate pref labels with human labelling
        achieved_state_1, achieved_state_2, goals, labels = self.generate_pref_labels()
        
        self.reward_model_buffer.add_multiple_data_points(achieved_state_1, achieved_state_2, goals, labels)

        # Train reward model
        losses_reward_model, eval_loss_reward_model = self.train_rewardmodel(batch_size=self.reward_model_batch_size)


        wandb.log({'LossesRewardModel/Train':np.mean(losses_reward_model), 'timesteps':self.timesteps})
            
        return losses_reward_model, eval_loss_reward_model
    
    def env_distance(self, state, goal):
        obs = self.exploration_env.observation(state)
        if "pointmass" in self.env_name:
            return self.exploration_env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.exploration_env.get_shaped_distance(obs, goal)
    
    def oracle_model(self, state):
        state = state.detach().cpu().numpy()

        goal = self.goal

        dist = [
            self.env_distance(state[i], goal) #+ np.random.normal(scale=self.distance_noise_std)
            for i in range(state.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array(dist)).reshape(-1,1)
        
        return scores

    def test_rewardmodel(self, itr):
        goal =self.goal#np.random.uniform(-0.5, 0.5, size=(2,))
        goal_pos =  goal
        #goal_pos = goal
        #TODO: remove
        #goal_pos = np.array([0.3,0.3])
        goals = np.repeat(goal_pos[None], 10000, axis=0)
        states = np.random.uniform(-0.6, 0.6, size=(10000, 2))
        states_t = torch.Tensor(states).cuda()
        goals_t = torch.Tensor(goals).cuda()
        r_val = self.reward_model(states_t)
        #print("goal pos", goal_pos.shape)
        #r_val = self.oracle_model(states_t, goals_t)
        r_val = r_val.cpu().detach().numpy()
        plt.clf()
        plt.cla()
        self.display_wall()
        plt.scatter(states[:, 0], states[:, 1], c=r_val[:, 0], cmap=cm.jet)


        plt.scatter(goal_pos[0], goal_pos[1], marker='o', s=100, color='black')

        
        plt.savefig("pointmass/rewardmodel_test/test_rewardmodel_itr%d.png"%itr)


    def train_rewardmodel(self, num_epochs=400, batch_size=32, device='cuda' ):
         # Train standard goal conditioned policy

        loss_fn = torch.nn.CrossEntropyLoss() 
        losses_eval = []

        self.reward_model.train()
        running_loss = 0.0
        
        # Train the model with regular SGD
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            start = time.time()
            
            achieved_states_1, achieved_states_2, goals ,labels = self.reward_model_buffer.sample_batch(batch_size)
            
            self.reward_optimizer.zero_grad()

            t_idx = np.random.randint(len(goals), size=(batch_size,)) # Indices of first trajectory
            
            state1 = torch.Tensor(achieved_states_1[t_idx]).to(device)
            state2 = torch.Tensor(achieved_states_2[t_idx]).to(device)
            goal = torch.Tensor(goals[t_idx]).to(device)
            label_t = torch.Tensor(labels[t_idx]).long().to(device)

            g1g2 = torch.cat([self.reward_model(state1), self.reward_model(state2)], axis=-1)
            loss = loss_fn(g1g2, label_t)
            loss.backward()
            self.reward_optimizer.step()

            # print statistics
            running_loss += float(loss.item())

        return running_loss, 0
    
    def display_wall(self):
        walls = self.exploration_env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')
    def display_trajectories(self, trajs, iter):
        # plot added trajectories to fake replay buffer
        plt.clf()
        self.display_wall()
        
        colors = sns.color_palette('hls', (len(trajs)))
        for j in range(len(trajs)):
            color = colors[j]
            traj_state = trajs[j]['observations']
            plt.plot(traj_state[:,0], traj_state[:, 1], color=color, zorder = -1)
            #if 'train_states_preferences' in filename:
            #    color = 'black'
            
            plt.scatter(self.goal[-2],
                    self.goal[-1], marker='o', s=20, color=color, zorder=1)
        
        plt.savefig(f"pointmass/rewardmodel_test/traj_{iter}.png")

    def _train(self):
        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            if not self.offline_rl:
                self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)
            self.timesteps += self.min_num_steps_before_training
            rewards, distances = self.get_avg_stats(init_expl_paths)
            wandb.log({'Train/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})
            print({'Train/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})


        eval_paths = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )

        self.timesteps += self.num_eval_steps_per_epoch
        rewards, distances = self.get_avg_stats(eval_paths)
        wandb.log({'Eval/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})
        print({'Eval/Success': np.mean(rewards), 'Eval/Distance':np.mean(distances), 'timesteps':self.timesteps})
        """
                orig_reward = train_data['reward']
                wandb.log({
                    'Train/Success': train_data['reward'].sum(),
                    'RewardModel/Reward': train_data['reward']
                })
        """

        gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            
            print("here", self.num_expl_steps_per_train_loop, self.max_path_length)

            self.timesteps += self.num_expl_steps_per_train_loop
            eval_rewards, eval_distances = self.get_avg_stats(eval_paths)
            rewards, distances = self.get_avg_stats(new_expl_paths)
            if self.display_plots:
                self.display_trajectories(new_expl_paths, self.epoch)
            wandb.log({'Eval/Success': np.mean(eval_rewards), 'Eval/Distance':np.mean(eval_distances), 'timesteps':self.timesteps})
            wandb.log({'Train/Success': np.mean(rewards), 'Train/Distance':np.mean(distances), 'timesteps':self.timesteps})
            print({'Eval/Success': np.mean(eval_rewards), 'Eval/Distance':np.mean(eval_distances), 'timesteps':self.timesteps})
            print({'Train/Success': np.mean(rewards), 'Train/Distance':np.mean(distances), 'timesteps':self.timesteps})
            
            #TODO: save info on exploration paths

            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)

            gt.stamp('data storing', unique=False)
            if not self.use_oracle and self.epoch % self.train_rewardmodel_freq == 0:
                self.collect_and_train_rewardmodel(self.epoch)
                if self.display_plots:
                    self.test_rewardmodel(self.epoch)
            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                
                #goal_array_shape = (len(train_data),)+self.exploration_env.goal_space.shape
                #goal_array = np.zeros(goal_array_shape) + self.exploration_env.goal
                #goal_tensor = torch.tensor(goal_array)
                tensor_states = torch.tensor(train_data['next_observations']).float().to('cuda')
                if not self.use_oracle:
                    self.reward_model.eval()
                assert 'rewards' in train_data
                train_data['rewards'] = self.reward_model(tensor_states).detach().cpu().numpy()
                mean = np.mean(train_data['rewards'])
                std = np.std(train_data['rewards'])
                #train_data['rewards'] = (train_data['rewards'] - mean)/std
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)
