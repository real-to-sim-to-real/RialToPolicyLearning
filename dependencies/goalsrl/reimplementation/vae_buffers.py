import numpy as np
from goalsrl.reimplementation import buffer

import rlutil.torch.pytorch_util as ptu
import rlutil.torch as torch
import tqdm

class VAEReplayBuffer:
    def __init__(self,
                vae,
                beta=1,
                batch_size=512,
            ):

        self.vae = vae
        self.vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        self.vae_beta = beta
        self.vae_batch_size = batch_size

    def optimize(self, n_steps=1):
        losses = []
        with tqdm.trange(n_steps, leave=True) as ranger:
            for _ in ranger:
                traj_idxs = np.random.choice(self.current_buffer_size, self.vae_batch_size)
                goal_idxs = 1 + np.random.choice(self.max_trajectory_length - 1, self.vae_batch_size)
                goals = self.env.extract_goal(self._states[traj_idxs, goal_idxs])
                goals_torch = torch.tensor(goals, dtype=torch.float32)

                self.vae_optimizer.zero_grad()
                loss, recon_loss, kl_loss = self.vae.loss(goals_torch, beta=self.vae_beta)
                loss.backward()
                self.vae_optimizer.step()
                losses.append([ptu.to_numpy(loss), ptu.to_numpy(recon_loss), ptu.to_numpy(kl_loss)])
                ranger.set_description(str(np.mean(np.array(losses), 0)))
        return np.mean(losses)


class SkewfitReplayBuffer(buffer.GoalWeightedReplayBuffer, VAEReplayBuffer):
    def __init__(self,
                env,
                max_trajectory_length,
                buffer_size,
                vae,
                beta=1,
                batch_size=512,
                num_steps_per_traj=10,
                imagine_horizon=True,
                ):

        buffer.ReplayBuffer.__init__(self, env, max_trajectory_length, buffer_size, imagine_horizon=imagine_horizon)
        VAEReplayBuffer.__init__(self, vae, beta=beta, batch_size=batch_size)
        # Just for logging
        self._probabilities = np.zeros(
            (buffer_size, max_trajectory_length),
            dtype=np.float32
        )
        self.vae_update_steps = num_steps_per_traj

    def add_trajectory(self, states, actions, desired_state):
        super().add_trajectory(states, actions, desired_state)
        if np.random.rand() < 0.25:
            self.optimize(self.vae_update_steps * 4)

    def _recompute_weights(self):
        T = self.max_trajectory_length

        old_probs = np.tile(np.arange(T), (self.current_buffer_size, 1)).flatten()
        old_probs = old_probs / old_probs.sum()

        if self.current_buffer_size < 200: # 5000 steps
            self.official_weights = old_probs
            return

        for i in tqdm.trange(self.current_buffer_size):
            goals = self.env.extract_goal(self._states[i])
            goals_torch = torch.tensor(goals, dtype=torch.float32)
            self._probabilities[i] = ptu.to_numpy(self.vae.marginal_prob(goals_torch, n=10, beta=self.vae_beta))

        modifiers = self._probabilities[:self.current_buffer_size].flatten()
        if np.max(modifiers) == 0:
            self.official_weights = old_probs
            return

        min_nonzero = modifiers[modifiers.nonzero()].min()
        modifiers = np.clip(modifiers, min_nonzero, 1e10)
        new_probs = np.nan_to_num(1 / modifiers) ** 0.6
        self.official_weights = new_probs / np.sum(new_probs)

    def state_dict(self):
        """
        To be saved in buffer.pkl
        """
        d = super().state_dict()
        d['vae'] = self.vae
        d['probabilities'] = self._probabilities[:self.current_buffer_size]
        return d

def create_bin_fn(vae, granularity, latent_dim):
    def bin_fn(goals):
        goals_torch = torch.tensor(goals, dtype=torch.float32)
        latents_torch = vae.get_latents(goals_torch)
        latents = np.clip(ptu.to_numpy(latents_torch), -2, 2)
        differents = np.linspace(-2, 2, granularity)
        bin_counts = granularity ** np.arange(latents.shape[1])
        return (np.argmax(np.tile(latents[..., None], (1, 1, granularity)) < differents, axis=-1) * bin_counts).sum(1)

    return bin_fn, granularity ** latent_dim

class VAEBinReplayBuffer(buffer.BinWeightedReplayBuffer, VAEReplayBuffer):
    def __init__(
                self,
                env,
                max_trajectory_length,
                buffer_size,
                vae,
                latent_dim=None,
                beta=1,
                batch_size=512,
                num_steps_per_traj=10,
                granularity=10,
                imagine_horizon=True,
                ):

        VAEReplayBuffer.__init__(self, vae, beta=beta, batch_size=batch_size)
        bin_fn, n_bins = create_bin_fn(vae, granularity, latent_dim)
        buffer.BinWeightedReplayBuffer.__init__(self, env, max_trajectory_length, buffer_size,
            reweight_fn=bin_fn, n_reweight_bins=n_bins, use_internal_goals=False)
        self.vae_update_steps = num_steps_per_traj * 10
        self.updated_vae = True

    def add_trajectory(self, states, actions, desired_state):
        super().add_trajectory(states, actions, desired_state)
        if np.random.rand() < 0.10:
            self.optimize(self.vae_update_steps)
            self.updated_vae = True

    def _recompute_weights(self):
        if self.updated_vae: # At least 1000 new steps added
            self.refresh_bins()
            self.updated_vae = False
        super()._recompute_weights()

    def state_dict(self):
        """
        To be saved in buffer.pkl
        """
        d = super().state_dict()
        d['vae'] = self.vae
        return d
