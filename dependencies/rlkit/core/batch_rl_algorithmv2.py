import abc

import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector

import numpy as np
import torch
import time
import wandb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from PIL import Image


class BatchRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
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
        self.goal = exploration_env.goal

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
    def display_wall(self):
        walls = self.exploration_env.base_env.room.get_walls()
        for wall in walls:
            start, end = wall
            sx, sy = start
            ex, ey = end
            plt.plot([sx, ex], [sy, ey], marker='o',  color = 'b')

    def display_trajectories(self, trajs, iter=0):
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
        filename = f"pointmass/rewardmodel_test/traj_{iter}.png"
        plt.savefig(filename)
        image = Image.open(filename)
        image = np.asarray(image)[:,:,:3]
        images = wandb.Image(image, caption="Top: Output, Bottom: Input")

        wandb.log({"plot_trajectories": images})

    def env_distance(self, state, goal):
        obs = self.exploration_env.observation(state)
        self.env_name = "pointmass_rooms"
        if "pointmass" in self.env_name:
            return self.exploration_env.base_env.room.get_shaped_distance(obs, goal)
        else:
            return self.exploration_env.get_shaped_distance(obs, goal)
    
    def oracle_model(self, state, goal):

        dist = [
            self.env_distance(state[i], goal) #+ np.random.normal(scale=self.distance_noise_std)
            for i in range(state.shape[0])
        ] #- np.linalg.norm(state - goal, axis=1)

        scores = - torch.tensor(np.array(dist)).reshape(-1,1)
        
        return scores
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

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
                discard_incomplete_paths=False,
            )
            
            self.display_trajectories(new_expl_paths)

            gt.stamp('exploration sampling', unique=False)

            if not self.offline_rl:
                self.replay_buffer.add_paths(new_expl_paths)
            gt.stamp('data storing', unique=False)

            self.training_mode(True)
            for _ in range(self.num_trains_per_train_loop):
                train_data = self.replay_buffer.random_batch(self.batch_size)
                tensor_states = torch.tensor(train_data['next_observations']).float().to('cuda')

                train_data['rewards'] = self.oracle_model(tensor_states.cpu().numpy(), self.goal).cpu().numpy()
                self.trainer.train(train_data)
            gt.stamp('training', unique=False)
            self.training_mode(False)