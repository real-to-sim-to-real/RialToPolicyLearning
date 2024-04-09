import numpy as np

class ReplayBuffer:
    """
    The base class for a replay buffer: stores goalsrl.envs.GoalEnv states,
    and on sampling time, chooses out the observation, goals, etc using the 
    env.observation, etc class

    Use the `imagine_horizon` argument to relabel horizons h, as any in {h, h+1, ... T}
    """

    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                imagine_horizon=False,
                ):
        """
        Args:
            env: A goalsrl.envs.GoalEnv
            max_trajectory_length (int): The length of each trajectory (must be fixed)
            buffer_size (int): The maximum number of trajectories in the buffer
            imagine_horizon (bool): If True, a horizon `h` will be randomly relabelled
                to arbitrary in {h, h+1, h+2, ... T}
        """
        self.env = env
        self._actions = np.zeros(
            (buffer_size, max_trajectory_length, *env.action_space.shape),
            dtype=env.action_space.dtype
        )
        self._states = np.zeros(
            (buffer_size, max_trajectory_length, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        self._desired_states = np.zeros(
            (buffer_size, *env.state_space.shape),
            dtype=env.state_space.dtype
        )
        
        internal_goal_shape = env._extract_sgoal(env.sample_goal()).shape
        self._internal_goals = np.zeros(
            (buffer_size, max_trajectory_length, *internal_goal_shape),
            dtype=env.observation_space.dtype,
        )
        
        self._length_of_traj = np.zeros(
            (buffer_size,),
            dtype=int
        )
        self.pointer = 0
        self.current_buffer_size = 0
        self.max_buffer_size = buffer_size
        self.max_trajectory_length = max_trajectory_length
        self.added_since_sampled = True # DIRTY BIT
        
        self.imagine_horizon = imagine_horizon

    def add_trajectory(self, states, actions, desired_state, length_of_traj=None):
        """
        Adds a trajectory to the buffer

        Args:
            states (np.array): Environment states witnessed - Needs shape (self.max_path_length, *state_space.shape)
            actions (np.array): Actions taken - Needs shape (max_path_length, *action_space.shape)
            desired_state (np.array): The state attempting to be reached - Needs shape state_space.shape
        
        Returns:
            None
        """

        assert actions.shape == self._actions[0].shape
        assert states.shape == self._states[0].shape

        self._actions[self.pointer] = actions
        self._states[self.pointer] = states
        self._internal_goals[self.pointer] = self.env._extract_sgoal(states)
        self._desired_states[self.pointer] = desired_state
        if length_of_traj is None:
            length_of_traj = self.max_trajectory_length
        self._length_of_traj[self.pointer] = length_of_traj

        self.pointer += 1
        self.current_buffer_size = max(self.pointer, self.current_buffer_size)
        self.pointer %= self.max_buffer_size
        self.added_since_sampled = True # DIRTY BIT
    
    def _sample_indices(self, batch_size):
        traj_idxs = np.random.choice(self.current_buffer_size, batch_size)

        prop_idxs_1 = np.random.rand(batch_size) # (self.max_trajectory_length - 1, batch_size)
        prop_idxs_2 = np.random.rand(batch_size)
        time_idxs_1 = np.floor(prop_idxs_1 * (self._length_of_traj[traj_idxs]-1)).astype(int)
        time_idxs_2 = np.floor(prop_idxs_2 * (self._length_of_traj[traj_idxs])).astype(int)
        # time_idxs_2 = (np.ones(batch_size) * (self.max_trajectory_length - 1)).astype(int)
        # time_idxs_1 = np.random.choice(self.max_trajectory_length - 1, batch_size)
        # time_idxs_2 = np.random.choice(self.max_trajectory_length, batch_size)
        time_idxs_2[time_idxs_1 == time_idxs_2] += 1

        time_state_idxs = np.minimum(time_idxs_1, time_idxs_2)
        time_goal_idxs = np.maximum(time_idxs_1, time_idxs_2)
        return traj_idxs, time_state_idxs, time_goal_idxs

    def sample_batch(self, batch_size):
        """
        Samples a batch of data
        
        Args:
            batch_size (int): The size of the batch to be sampled
        Returns:
            observations
            actions
            goals
            lengths - Distance between observations and goals
            horizons - Lengths in reverse temperature encoding: if length=3, (0,0,0,1,1,1,1,1,1...)
            weights - Will be all ones (uniform)
        """

        traj_idxs, time_state_idxs, time_goal_idxs = self._sample_indices(batch_size)
        self.added_since_sampled = False
        return self._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        batch_size = len(traj_idxs)
        observations = self.env.observation(self._states[traj_idxs, time_state_idxs])
        actions = self._actions[traj_idxs, time_state_idxs]
        goals = self.env.extract_goal(self._states[traj_idxs, time_goal_idxs])
        
        lengths = time_goal_idxs - time_state_idxs
        
        if self.imagine_horizon:
            lengths2 = lengths + np.random.rand(batch_size) * (self.max_trajectory_length - lengths)
            lengths = np.where(np.random.rand(batch_size) < 2/3, lengths, lengths2)

        horizons = np.tile(np.arange(self.max_trajectory_length), (batch_size, 1))
        horizons = horizons >= lengths[..., None]

        weights = np.ones(batch_size)

        return observations, actions, goals, lengths, horizons, weights
    
    def save(self, file_name):
        np.savez(file_name,
            states=self._states[:self.current_buffer_size],
            actions=self._actions[:self.current_buffer_size],
            desired_states=self._desired_states[:self.current_buffer_size],
        )

    def load(self, file_name, replace=False):
        data = np.load(file_name)
        states, actions, desired_states = data['states'], data['actions'], data['desired_states']
        n_trajectories = len(states)
        for i in range(n_trajectories):
            self.add_trajectory(states[i], actions[i], desired_states[i])

    def state_dict(self):
        d = dict(internal_goals=self._internal_goals[:self.current_buffer_size])
        if self._states.shape[2] < 100: # Not images
            d.update(dict(
                states=self._states[:self.current_buffer_size],
                actions=self._actions[:self.current_buffer_size],
                desired_states=self._desired_states[:self.current_buffer_size],
            ))
        return d

class GoalWeightedReplayBuffer(ReplayBuffer):
    """
    Samples goals uniformly according to some probability distribution on goals
    """
    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                **kwargs
        ):
        super(GoalWeightedReplayBuffer, self).__init__(env, max_trajectory_length, buffer_size, **kwargs)
        self.official_weights = None
        
    def _sample_indices(self, batch_size):
        if self.added_since_sampled:
            self._recompute_weights()
            self.added_since_sampled = False
        
        full_idxs = np.random.choice(len(self.official_weights), batch_size, p=self.official_weights)
        traj_idxs = (full_idxs // self.max_trajectory_length).astype(int)
        time_goal_idxs = (full_idxs % self.max_trajectory_length).astype(int)
        time_state_idxs = np.floor(np.random.rand(batch_size) * time_goal_idxs).astype(int)
        
        return traj_idxs, time_state_idxs, time_goal_idxs
    
    def _recompute_weights(self):
        original_weights = np.tile(
            np.arange(self.max_trajectory_length),
            (self.current_buffer_size, 1)
        )
        self.official_weights = original_weights.flat[:] / original_weights.sum()

    def state_dict(self):
        """
        To be saved in buffer.pkl
        """
        d = super().state_dict()
        d['official_weights'] = self.official_weights
        return d

class BinWeightedReplayBuffer(GoalWeightedReplayBuffer):
    """
    Samples goals uniformly according to some binning function on goals or internal goals
    """
    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                # Reweighting scheme
                reweight_fn=None,
                n_reweight_bins=0,
                use_internal_goals=True,
                **kwargs
                ):

        super(BinWeightedReplayBuffer, self).__init__(env, max_trajectory_length, buffer_size, **kwargs)
        self._bins = np.zeros((buffer_size, max_trajectory_length), dtype=np.int)
        self.reweight_fn = reweight_fn
        self.n_reweight_bins = n_reweight_bins
        self.use_internal_goals = use_internal_goals
    
    def add_trajectory(self, states, actions, desired_state):

        if self.use_internal_goals:
            self._bins[self.pointer] = self.reweight_fn(self.env._extract_sgoal(states))
        else:
            self._bins[self.pointer] = self.reweight_fn(self.env.extract_goal(states))
        
        super().add_trajectory(states, actions, desired_state)
    
    def refresh_bins(self):
        for i in range(self.current_buffer_size):
            if self.use_internal_goals:
                self._bins[i] = self.reweight_fn(self._internal_goals[i])
            else:
                self._bins[i] = self.reweight_fn(self.env.extract_goal(self._states[i]))

    def _recompute_weights(self):
        T = self.max_trajectory_length
        
        counts = np.zeros(self.n_reweight_bins)
        proper_counts = np.zeros(self.n_reweight_bins)

        bins = self._bins[:self.current_buffer_size]
        
        for traj in range(self.current_buffer_size):
            for g in range(T):
                counts[bins[traj][g]] += g+1
                proper_counts[bins[traj][g]] += 1
        
        print(counts)

        original_weights = np.tile(
            np.arange(T),
            (self.current_buffer_size, 1)
        )
        # original_weights = np.tile(
        #     np.arange(T) > 0,
        #     (self.current_buffer_size, 1)
        # ).astype(int)

        unnorm_weights = np.nan_to_num(original_weights / counts[bins])
        unnorm_weights = np.clip(unnorm_weights.flat[:], 0, 1)        
        self.official_weights = unnorm_weights / np.sum(unnorm_weights)

    def state_dict(self):
        """
        To be saved in buffer.pkl
        """
        d = super().state_dict()
        d['bins'] = self._bins[:self.current_buffer_size]
        return d