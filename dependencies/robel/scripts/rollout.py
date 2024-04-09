# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to perform rollouts on an environment.

Example usage:
# Visualize an environment:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render

# Benchmark offscreen rendering:
python -m robel.scripts.rollout -e DClawTurnFixed-v0 --render rgb_array
"""

import argparse
import collections
import os
import pickle
import time
from typing import Callable, Optional

import gym
import numpy as np

import robel
from robel.scripts.utils import EpisodeLogger, parse_env_args

# The default environment to load when no environment name is given.
DEFAULT_ENV_NAME = 'DClawTurnFixed-v0'

# The default number of episodes to run.
DEFAULT_EPISODE_COUNT = 10

# Named tuple for information stored over a trajectory.
Trajectory = collections.namedtuple('Trajectory', [
    'actions',
    'observations',
    'rewards',
    'total_reward',
    'infos',
    'renders',
    'durations',
])


def do_rollouts(env,
                num_episodes: int,
                max_episode_length: Optional[int] = None,
                action_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                render_mode: Optional[str] = None):
    """Performs rollouts with the given environment.

    Args:
        num_episodes: The number of episodes to do rollouts for.
        max_episode_length: The maximum length of an episode.
        action_fn: The function to use to sample actions for steps. If None,
            uses random actions.
        render_mode: The rendering mode. If None, no rendering is performed.

    Yields:
        Trajectory containing:
            observations: The observations for the episode.
            rewards: The rewards for the episode.
            total_reward: The total reward during the episode.
            infos: The auxiliary information during the episode.
            renders: Rendered frames during the episode.
            durations: The running execution durations.
    """
    # If no action function is given, use random actions from the action space.
    if action_fn is None:
        action_fn = lambda _: env.action_space.sample()

    # Maintain a dictionary of execution durations.
    durations = collections.defaultdict(float)

    # Define a function to maintain a running average of durations.
    def record_duration(key: str, iteration: int, value: float):
        durations[key] = (durations[key] * iteration + value) / (iteration + 1)

    total_steps = 0
    for episode in range(num_episodes):
        episode_start = time.time()
        obs = env.reset()
        record_duration('reset', episode, time.time() - episode_start)

        done = False
        episode_actions = []
        episode_obs = [obs]
        episode_rewards = []
        episode_total_reward = 0
        episode_info = collections.defaultdict(list)
        episode_renders = []

        while not done:
            step_start = time.time()

            # Get the action for the current observation.
            action = action_fn(obs)
            action_time = time.time()
            record_duration('action', total_steps, action_time - step_start)

            # Advance the environment with the action.
            obs, reward, done, info = env.step(action)
            step_time = time.time()
            record_duration('step', total_steps, step_time - action_time)

            # Render the environment if needed.
            if render_mode is not None:
                render_result = env.render(render_mode)
                record_duration('render', total_steps, time.time() - step_time)
                if render_result is not None:
                    episode_renders.append(render_result)

            # Record episode information.
            episode_actions.append(action)
            episode_obs.append(obs)
            episode_rewards.append(reward)
            episode_total_reward += reward
            for key, value in info.items():
                episode_info[key].append(value)

            total_steps += 1
            if (max_episode_length is not None
                    and len(episode_obs) >= max_episode_length):
                done = True

        # Combine the information into a trajectory.
        trajectory = Trajectory(
            actions=np.array(episode_actions),
            observations=np.array(episode_obs),
            rewards=np.array(episode_rewards),
            total_reward=episode_total_reward,
            infos={key: np.array(value) for key, value in episode_info.items()},
            renders=np.array(episode_renders) if episode_renders else None,
            durations=dict(durations),
        )
        yield trajectory


def rollout_script(arg_def_fn=None,
                   env_factory=None,
                   policy_factory=None,
                   add_policy_arg: bool = False):
    """Performs a rollout script.

    Args:
        arg_def_fn: A function that takes an ArgumentParser. Use this to add
            arguments to the script.
        env_factory: A function that takes program arguments and returns
            an environment. Otherwise, uses `gym.make`.
        policy_factory: A function that takes program arguments and returns a
            policy function (callable that observations and returns actions)
            and the environment.
        add_policy_arg: If True, adds an argument to take a policy path.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', help='The directory to save rollout data to.')
    if add_policy_arg:
        parser.add_argument(
            '-p', '--policy', help='The path to the policy file to load.')
    parser.add_argument(
        '-n',
        '--num_episodes',
        type=int,
        default=DEFAULT_EPISODE_COUNT,
        help='The number of episodes to run.')
    parser.add_argument(
        '--seed', type=int, default=None, help='The seed for the environment.')
    parser.add_argument(
        '-r',
        '--render',
        nargs='?',
        const='human',
        default=None,
        help=('The rendering mode. If provided, renders to a window. A render '
              'mode string can be passed here.'))
    # Add additional argparse arguments.
    if arg_def_fn:
        arg_def_fn(parser)
    env_id, params, args = parse_env_args(
        parser, default_env_name=DEFAULT_ENV_NAME)

    robel.set_env_params(env_id, params)
    if env_factory:
        env = env_factory(args)
    else:
        env = gym.make(env_id)

    action_fn = None
    if policy_factory:
        action_fn = policy_factory(args)

    if args.seed is not None:
        env.seed(args.seed)

    paths = []
    try:
        episode_num = 0
        for traj in do_rollouts(
                env,
                num_episodes=args.num_episodes,
                action_fn=action_fn,
                render_mode=args.render,
        ):
            print('Episode {}'.format(episode_num))
            print('> Total reward: {}'.format(traj.total_reward))
            if traj.durations:
                print('> Execution times:')
                for key in sorted(traj.durations):
                    print('{}{}: {:.2f}ms'.format(' ' * 4, key,
                                                  traj.durations[key] * 1000))
            episode_num += 1

            if args.output:
                paths.append(
                    dict(
                        actions=traj.actions,
                        observations=traj.observations,
                        rewards=traj.rewards,
                        total_reward=traj.total_reward,
                        infos=traj.infos,
                    ))
    finally:
        env.close()

        if paths and args.output:
            os.makedirs(args.output, exist_ok=True)
            # Serialize the paths.
            save_path = os.path.join(args.output, 'paths.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(paths, f)

            # Log the paths to a CSV file.
            csv_path = os.path.join(args.output,
                                    '{}-results.csv'.format(env_id))
            with EpisodeLogger(csv_path) as logger:
                for path in paths:
                    logger.log_path(path)


if __name__ == '__main__':
    # If calling this script directly, do rollouts with a random policy.
    rollout_script()
