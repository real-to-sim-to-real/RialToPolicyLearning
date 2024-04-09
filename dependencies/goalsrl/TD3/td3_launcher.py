import numpy as np
import torch
import gym
import argparse
import os.path as osp
import pickle

from goalsrl.TD3 import utils
from goalsrl.TD3 import TD3
from gym.wrappers.time_limit import TimeLimit
from rlutil.logging import logger
import wandb
import time
import tqdm
# Runs policy for X episodes and returns average reward


class TD3Algo:
    def __init__(self,
        env,
        policy,
        max_path_length=50,
        goal_threshold=0.05,
        max_timesteps=1e6,
        buffer_size=None,
        prob_self_goal=0.3,
        start_timesteps=1e4,
        eval_freq=5e3,
        eval_episodes=200,
        expl_noise=0.1,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        lr=1e-3,
        critic_updates_per_step=1,
        policy_freq=2,
    ):

        self.env = env
        self.policy = policy # TD3.TD3(self.env, actor_kwargs=actor_kwargs, critic_kwargs=critic_kwargs, lr=lr)
        
        if buffer_size is None:
            buffer_size = max_timesteps
        
        self.replay_buffer = utils.ReplayBuffer(
            max_size=buffer_size,
            max_lookahead=max_path_length,
            prob_self_goal=prob_self_goal
        )

        self.max_path_length = max_path_length
        self.goal_threshold = goal_threshold
        self.start_timesteps = start_timesteps
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.max_timesteps = max_timesteps
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.prob_self_goal = prob_self_goal
        self.lr = lr
        self.critic_updates_per_step = critic_updates_per_step
        

    def train(self):
        env = self.env
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        self.total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True 
        self.evaluate_policy(eval_episodes=self.eval_episodes)
        logger.record_tabular('timesteps', self.total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        last_time = time.time()
        logger.dump_tabular()
                    
        
        with tqdm.tqdm(total=self.eval_freq, leave=True) as ranger:
            while self.total_timesteps < self.max_timesteps:
                if done: 
                    if self.total_timesteps != 0: 
                        # print(("Total T: %d Episode Num: %d Episode T: %d Avg Distance: %f Final Distance: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward / 50, reward))
                        if self.critic_updates_per_step >= 1:
                            for _ in range(self.critic_updates_per_step):
                                self.policy.train(self.replay_buffer, episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)
                        else:
                            self.policy.train(self.replay_buffer, int(self.critic_updates_per_step * episode_timesteps), self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)

                    # Evaluate episode
                    if timesteps_since_eval >= self.eval_freq:
                        timesteps_since_eval %= self.eval_freq
                        self.evaluate_policy(eval_episodes=self.eval_episodes)
                        logger.record_tabular('timesteps', self.total_timesteps)
                        logger.record_tabular('epoch time (s)', time.time() - last_time)
                        logger.record_tabular('total time (s)', time.time() - start_time)
                        last_time = time.time()
                        logger.dump_tabular()
                        if logger.get_snapshot_dir():
                            self.policy.save('policy', logger.get_snapshot_dir())
                            full_dict = dict(env=self.env, policy=self.policy)
                            with open(osp.join(logger.get_snapshot_dir(), 'params.pkl'), 'wb') as f:
                                pickle.dump(full_dict, f)

                        ranger.reset()
                    
                    # Reset environment
                    goal_state = env.sample_goal()
                    goal = env.extract_goal(goal_state)

                    state = env.reset()
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1 
                
                # Select action randomly or according to policy
                if self.total_timesteps < self.start_timesteps:
                    action = env.action_space.sample()
                else:
                    obs = env.observation(state)
                    action = self.policy.select_action(obs, goal)
                    if self.expl_noise != 0: 
                        if hasattr(self.env.action_space, 'n'): # discrete
                            if np.random.rand() < self.expl_noise:
                                action = self.env.action_space.sample()
                        else:   
                            action = (action + np.random.normal(0, self.expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

                # Perform action
                new_state, _, done, _ = env.step(action) 
                reward = env.goal_distance(new_state, goal_state)

                done = 1 if episode_timesteps + 1 == self.max_path_length else float(done)
                episode_reward += reward

                # Store data in replay buffer
                self.replay_buffer.add((env.observation(state), env.extract_goal(state), env.observation(new_state), action, goal, reward, done))

                state = new_state

                episode_timesteps += 1
                self.total_timesteps += 1
                timesteps_since_eval += 1
                ranger.update(1)
    
    def sample_trajectory(self, noise=0):
        
        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)

        states = []
        actions = []

        state = self.env.reset()
        for t in range(self.max_path_length):
            states.append(state)

            observation = self.env.observation(state)
            action = self.policy.select_action(observation, goal)
            if hasattr(self.env.action_space, 'n'):
                if np.random.rand() < noise: # Discrete hack
                    action = self.env.action_space.sample()
            else:
                action += np.random.randn(*action.shape)  * noise
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            actions.append(action)
            state, _, _, _ = self.env.step(action)
        
        return np.stack(states), np.array(actions), goal_state

    def evaluate_policy(self, eval_episodes=200, prefix='EvalGreedy'):
        env = self.env

        all_states = []
        all_goal_states = []

        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state = self.sample_trajectory(noise=0)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.compute_shaped_distance(env.observation(states[-1]), env.extract_goal(goal_state))
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes'%prefix, eval_episodes)
        logger.record_tabular('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio'%prefix, np.mean(success_vec))

        wandb.log({'Eval/avg final dist':np.mean(final_dist_vec), 'Eval/success ratio':np.mean(success_vec), 'timesteps':self.total_timesteps})
        """
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        
        """
        
        return
