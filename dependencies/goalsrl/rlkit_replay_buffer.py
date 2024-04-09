import numpy as np
from gym.spaces import Dict, Discrete

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer


class GoalsRLRelabelingBuffer(ObsDictRelabelingBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_next_goals=0.4,
            fraction_trajectory_goals=0.5,
            fraction_rollout_goals=0.1,
            #fraction_goals_rollout_goals=1.0,
            #fraction_goals_env_goals=0.0,
            internal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
    ):
        super(GoalsRLRelabelingBuffer, self).__init__(
            max_size, env,
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
            internal_keys=internal_keys,
            goal_keys=goal_keys,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
            achieved_goal_key=achieved_goal_key,
        )

        self.fraction_next_goals = fraction_next_goals
        self.fraction_trajectory_goals = fraction_trajectory_goals
        self.fraction_rollout_goals = fraction_rollout_goals
        assert (fraction_rollout_goals + fraction_next_goals + fraction_trajectory_goals) <= 1.0


    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_next_goals = int(batch_size * self.fraction_next_goals)
        num_trajectory_goals = int(batch_size * self.fraction_trajectory_goals)
        num_rollout_goals = batch_size - (num_next_goals + num_trajectory_goals)
        
        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_trajectory_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_trajectory_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            resampled_goals[-num_trajectory_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_trajectory_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_trajectory_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        if num_next_goals > 0:
            future_obs_idxs = indices[:num_next_goals] + 1
            resampled_goals[:num_next_goals] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][:num_next_goals] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][:num_next_goals] = \
                    self._next_obs[goal_key][future_obs_idxs]
            
        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        new_actions = self._actions[indices]
        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """
        new_rewards = np.zeros((batch_size, 1))
        new_rewards[:num_next_goals] = 1
        new_rewards = new_rewards.reshape(-1, 1)

        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        terminals = np.zeros((batch_size, 1), dtype='uint8')
        terminals[:num_next_goals] = 1

        batch = {
            'observations': new_obs,
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': terminals,
            'next_observations': new_next_obs,
            'resampled_goals': resampled_goals,
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0
