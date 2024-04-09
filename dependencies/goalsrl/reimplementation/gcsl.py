import numpy as np
from rlutil.logging import logger

import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu

import time
import tqdm
import os.path as osp
import copy
import pickle
try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_enabled = True
except:
    print('Tensorboard not installed!')
    tensorboard_enabled = False

class GCSL:
    def __init__(self,
        env,
        policy,
        replay_buffer,
        validation_buffer=None,
        demo_replay_buffer=None,
        demo_validation_replay_buffer=None,
        max_path_length=50,
        goal_threshold=0.05,
        demo_train_steps=0,
        start_timesteps=1e4,
        start_policy_timesteps=0,
        eval_freq=5e3,
        eval_episodes=200,
        save_every_iteration=False,
        max_timesteps=1e6,
        expl_noise=0.1,
        batch_size=100,
        n_accumulations=1,
        policy_updates_per_step=1,
        reset_policy_freq=float('inf'),
        train_policy_freq=None,
        lr=5e-4,
        log_tensorboard=False,
    ):

        self.env = env
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.validation_buffer = validation_buffer
        self.demo_replay_buffer = demo_replay_buffer
        self.demo_validation_replay_buffer = demo_validation_replay_buffer

        self.is_discrete_action = hasattr(self.env.action_space, 'n')

        self.max_path_length = max_path_length
        self.goal_threshold = goal_threshold
        self.start_timesteps = start_timesteps
        self.start_policy_timesteps = start_policy_timesteps

        self.demo_train_steps = demo_train_steps

        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_every_iteration = save_every_iteration

        self.max_timesteps = max_timesteps
        self.expl_noise = expl_noise

        if train_policy_freq is None:
            train_policy_freq = self.max_path_length

        self.train_policy_freq = train_policy_freq
        self.batch_size = batch_size
        self.n_accumulations = n_accumulations
        self.policy_updates_per_step = policy_updates_per_step
        
        self.reset_policy_freq = reset_policy_freq
        self.original_policy_dict = copy.deepcopy(self.policy.state_dict())
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.log_tensorboard = log_tensorboard and tensorboard_enabled
        self.summary_writer = None


    def loss_fn(self, observations, goals, actions, horizons, weights):
        obs_dtype = torch.float32
        action_dtype = torch.int64 if self.is_discrete_action else torch.float32

        observations_torch = torch.tensor(observations, dtype=obs_dtype)
        goals_torch = torch.tensor(goals, dtype=obs_dtype)
        actions_torch = torch.tensor(actions, dtype=action_dtype)
        horizons_torch = torch.tensor(horizons, dtype=obs_dtype)
        weights_torch = torch.tensor(weights, dtype=torch.float32)
        nll = self.policy.nll(observations_torch, goals_torch, actions_torch, horizon=horizons_torch)

        return torch.mean(nll * weights_torch)
    
    def sample_trajectory(self, greedy=False, noise=0,):

        goal_state = self.env.sample_goal()
        goal = self.env.extract_goal(goal_state)

        states = []
        actions = []

        state = self.env.reset()
        for t in range(self.max_path_length):
            states.append(state)

            observation = self.env.observation(state)
            horizon = np.arange(self.max_path_length) >= (self.max_path_length - 1 - t)
            action = self.policy.act_vectorized(observation[None], goal[None], horizon=horizon[None], greedy=greedy, noise=noise)[0]
            
            if not self.is_discrete_action:
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            
            actions.append(action)
            state, _, _, _ = self.env.step(action)
        
        return np.stack(states), np.array(actions), goal_state

    def take_policy_step(self, demo=False):
        buffer = self.demo_replay_buffer if demo else self.replay_buffer
        avg_loss = 0
        self.policy_optimizer.zero_grad()
        
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            loss.backward()
            avg_loss += ptu.to_numpy(loss)
        
        # Adding Weight Decay
        # for group in self.policy_optimizer.param_groups:
        #     for param in group['params']:
        #         param.data = param.data.add(-0.3 * group['lr'], param.data)
        
        self.policy_optimizer.step()
        
        return avg_loss / self.n_accumulations

    def validation_loss(self, demo=False):
        buffer = self.demo_validation_replay_buffer if demo else self.validation_buffer

        if buffer is None or buffer.current_buffer_size == 0:
            return 0

        avg_loss = 0
        for _ in range(self.n_accumulations):
            observations, actions, goals, lengths, horizons, weights = buffer.sample_batch(self.batch_size)
            loss = self.loss_fn(observations, goals, actions, horizons, weights)
            avg_loss += ptu.to_numpy(loss)

        return avg_loss / self.n_accumulations

    def train(self):
        env = self.env
        start_time = time.time()
        last_time = start_time

        # Evaluate untrained policy
        total_timesteps = 0
        timesteps_since_train = 0
        timesteps_since_eval = 0
        timesteps_since_reset = 0

        iteration = 0
        episode_num = 0
        running_loss = None
        running_validation_loss = None
        losses = []
        if logger.get_snapshot_dir() and self.log_tensorboard:
            self.summary_writer = SummaryWriter(osp.join(logger.get_snapshot_dir(),'tensorboard'))

        self.policy.train()
        if self.demo_replay_buffer is not None:
            looper = tqdm.trange(self.demo_train_steps)
            for _ in looper:
                loss = self.take_policy_step(demo=True)
                validation_loss = self.validation_loss(demo=True)

                if running_loss is None:
                    running_loss = loss
                else:
                    running_loss = 0.99 * running_loss + 0.01 * loss
                if running_validation_loss is None:
                    running_validation_loss = validation_loss
                else:
                    running_validation_loss = 0.99 * running_validation_loss + 0.01 * validation_loss

                looper.set_description('Loss: %.03f Validation Loss: %.03f'%(running_loss, running_validation_loss))

        self.policy.eval()
        self.evaluate_policy(self.eval_episodes, greedy=False, prefix='Eval')
        self.evaluate_policy(self.eval_episodes, greedy=True, prefix='EvalGreedy')
        logger.record_tabular('policy loss', 0)
        logger.record_tabular('timesteps', total_timesteps)
        logger.record_tabular('epoch time (s)', time.time() - last_time)
        logger.record_tabular('total time (s)', time.time() - start_time)
        last_time = time.time()
        logger.dump_tabular()
        
        
        with tqdm.tqdm(total=self.eval_freq, smoothing=0) as ranger:
            while total_timesteps < self.max_timesteps:
                if total_timesteps < self.start_timesteps:
                    states, actions, goal_state = self.sample_trajectory(noise=1)
                else:
                    states, actions, goal_state = self.sample_trajectory(noise=self.expl_noise)
                if self.validation_buffer is not None and np.random.rand() < 0.2:
                    self.validation_buffer.add_trajectory(states, actions, goal_state)
                else:
                    self.replay_buffer.add_trajectory(states, actions, goal_state)
                total_timesteps += self.max_path_length

                timesteps_since_train += self.max_path_length
                timesteps_since_eval += self.max_path_length
                timesteps_since_reset += self.max_path_length
                
                ranger.update(self.max_path_length)
                
                if timesteps_since_train >= self.train_policy_freq and total_timesteps > self.start_policy_timesteps:
                    timesteps_since_train %= self.train_policy_freq
                    self.policy.train()
                    for _ in range(int(self.policy_updates_per_step * self.train_policy_freq)):
                        loss = self.take_policy_step()
                        validation_loss = self.validation_loss()
                        losses.append(loss)
                        if running_loss is None:
                            running_loss = loss
                        else:
                            running_loss = 0.9 * running_loss + 0.1 * loss

                        if running_validation_loss is None:
                            running_validation_loss = validation_loss
                        else:
                            running_validation_loss = 0.9 * running_validation_loss + 0.1 * validation_loss

                    self.policy.eval()
                    ranger.set_description('Loss: %s Validation Loss: %s'%(running_loss, running_validation_loss))

                    if self.summary_writer:
                        self.summary_writer.add_scalar('Losses/Train', running_loss, total_timesteps)
                        self.summary_writer.add_scalar('Losses/Validation', running_validation_loss, total_timesteps)

                if timesteps_since_eval >= self.eval_freq:
                    timesteps_since_eval %= self.eval_freq
                    iteration += 1

                    self.evaluate_policy(self.eval_episodes, greedy=False, prefix='Eval', total_timesteps=total_timesteps)
                    self.evaluate_policy(self.eval_episodes, greedy=True, prefix='EvalGreedy', total_timesteps=total_timesteps)

                    logger.record_tabular('policy loss', running_loss or 0) # Handling None case
                    logger.record_tabular('timesteps', total_timesteps)
                    logger.record_tabular('epoch time (s)', time.time() - last_time)
                    logger.record_tabular('total time (s)', time.time() - start_time)
                    last_time = time.time()
                    logger.dump_tabular()
                    
                    if logger.get_snapshot_dir():
                        modifier = str(iteration) if self.save_every_iteration else ''
                        torch.save(
                            self.policy.state_dict(),
                            osp.join(logger.get_snapshot_dir(), 'policy%s.pkl'%modifier)
                        )
                        if hasattr(self.replay_buffer, 'state_dict'):
                            with open(osp.join(logger.get_snapshot_dir(), 'buffer%s.pkl'%modifier), 'wb') as f:
                                pickle.dump(self.replay_buffer.state_dict(), f)

                        full_dict = dict(env=self.env, policy=self.policy)
                        with open(osp.join(logger.get_snapshot_dir(), 'params%s.pkl'%modifier), 'wb') as f:
                            pickle.dump(full_dict, f)
                    ranger.reset()
                
                if timesteps_since_reset >= self.reset_policy_freq:
                    timesteps_since_reset %= self.reset_policy_freq
                    self.policy.load_state_dict(self.original_policy_dict)
                    running_loss = None
                    running_validation_loss = None
                  
                    
    def evaluate_policy(self, eval_episodes=200, greedy=True, prefix='Eval', total_timesteps=0):
        env = self.env
        policy = self.policy
        
        all_states = []
        all_goal_states = []
        all_actions = []
        final_dist_vec = np.zeros(eval_episodes)
        success_vec = np.zeros(eval_episodes)

        for index in tqdm.trange(eval_episodes, leave=True):
            states, actions, goal_state = self.sample_trajectory(noise=0, greedy=greedy)
            all_actions.extend(actions)
            all_states.append(states)
            all_goal_states.append(goal_state)
            final_dist = env.goal_distance(states[-1], goal_state)
            
            final_dist_vec[index] = final_dist
            success_vec[index] = (final_dist < self.goal_threshold)

        all_states = np.stack(all_states)
        all_goal_states = np.stack(all_goal_states)

        logger.record_tabular('%s num episodes'%prefix, eval_episodes)
        logger.record_tabular('%s avg final dist'%prefix,  np.mean(final_dist_vec))
        logger.record_tabular('%s success ratio'%prefix, np.mean(success_vec))
        if self.summary_writer:
            self.summary_writer.add_scalar('%s/avg final dist'%prefix, np.mean(final_dist_vec), total_timesteps)
            self.summary_writer.add_scalar('%s/success ratio'%prefix,  np.mean(success_vec), total_timesteps)
        diagnostics = env.get_diagnostics(all_states, all_goal_states)
        for key, value in diagnostics.items():
            logger.record_tabular('%s %s'%(prefix, key), value)
        
        return all_states, all_goal_states
