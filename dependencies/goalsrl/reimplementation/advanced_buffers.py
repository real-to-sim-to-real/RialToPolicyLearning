import numpy as np
import tqdm
from goalsrl.reimplementation.buffer import GoalWeightedReplayBuffer
import rlutil.torch as torch
import rlutil.torch.pytorch_util as ptu

def get_ranks(arr):
    temp = arr.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = 1 + np.arange(len(arr))
    return ranks

class RNDReplayBuffer(GoalWeightedReplayBuffer):
    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                network_fn, # Should take in goals
                target_network_fn=None,
                p_update=1,
                ):
        super(RNDReplayBuffer, self).__init__(env, max_trajectory_length, buffer_size)
        self._errors = np.zeros((buffer_size, max_trajectory_length))

        self.model_to_optimize = network_fn()
        import copy
        self.init_state_dict = copy.deepcopy(self.model_to_optimize.state_dict())

        if target_network_fn is None:
            target_network_fn = network_fn
        
        self.random_model = target_network_fn()

        self.optimizer = torch.optim.Adam(self.model_to_optimize.parameters(), lr=5e-4)
        self.p_update = p_update

    def get_errors(self, states):
        goals = self.env.extract_goal(states)
        goals_torch = torch.tensor(goals, dtype=torch.float32)
        pred1 = self.model_to_optimize(goals_torch)
        pred2 = self.random_model(goals_torch)
        return torch.sum((pred1 - pred2) ** 2, -1)
    
    def take_step(self, states):
        errors = self.get_errors(states)
        self.optimizer.zero_grad()
        loss = torch.mean(errors)
        loss.backward()
        self.optimizer.step()
        return ptu.to_numpy(errors)

    def add_trajectory(self, states, actions, desired_state):
        if self.pointer % 500 == 0:
            self.model_to_optimize.load_state_dict(self.init_state_dict)
            print('Resetting model')

        errors = ptu.to_numpy(self.get_errors(states))
        self._errors[self.pointer] = errors
        super().add_trajectory(states, actions, desired_state)
        self.added_since_sampled = True

    def _train(self, n_steps=50):
        running_loss = None 
        with tqdm.trange(50, leave=False) as ranger:
            for i in ranger:
                to_update_trajs, _, to_update_goals = super().sample_indices(batch_size)
                errors = self.take_step(self._states[to_update_trajs, to_update_goals])
                self._errors[to_update_trajs, to_update_goals] = ptu.to_numpy(errors)
                if running_loss is None:
                    running_loss = np.mean(errors)
                else:
                    running_loss = 0.9 * running_loss + 0.1 * np.mean(errors)
                ranger.set_description('RND Error: %f'%running_loss)

    def _recompute_weights(self, batch_size):  
        running_loss = None 
        with tqdm.trange(50, leave=False) as ranger:
            for i in ranger:
                to_update_trajs, _, to_update_goals = super().sample_indices(batch_size)
                errors = self.take_step(self._states[to_update_trajs, to_update_goals])
                self._errors[to_update_trajs, to_update_goals] = ptu.to_numpy(errors)
                if running_loss is None:
                    running_loss = np.mean(errors)
                else:
                    running_loss = 0.9 * running_loss + 0.1 * np.mean(errors)
                ranger.set_description('RND Error: %f'%running_loss)
        
        original_weights = np.tile(
            np.arange(self.max_trajectory_length),
            (self.current_buffer_size, 1)
        )
        new_errors = self._errors[:self.current_buffer_size]
        error_ranks = 1 / get_ranks(new_errors.flat[:])

        unnorm_weights = original_weights.flatten() * error_ranks
        self.official_weights = unnorm_weights / np.sum(unnorm_weights)

    def sample_batch(self, batch_size):
        traj_idxs, time_state_idxs, time_goal_idxs = self.sample_indices(batch_size)
        errors = ptu.to_numpy(self.get_errors(self._states[traj_idxs, time_goal_idxs]))
        self._errors[traj_idxs, time_goal_idxs] = errors

        if False:
            to_update_trajs, _, to_update_goals = super().sample_indices(batch_size)
            errors = self.take_step(self._states[to_update_trajs, to_update_goals])
            self._errors[to_update_trajs, to_update_goals] = ptu.to_numpy(errors)
            loss = np.mean(errors)
            if np.random.rand() < 0.1:
                print(loss)
            
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)
    
    def state_dict(self):
        d = super().state_dict()
        errors = self._errors[:self.current_buffer_size]
        return dict(errors=errors, weights=self.official_weights)