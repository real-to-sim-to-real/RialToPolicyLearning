import numpy as np
from gym import Env, logger
from gym import error
from gym.spaces import Box


class Wrapper(Env):
    def __init__(self, env=None):
        self._wrapped_env = env

    def __getattr__(self, key):
        return getattr(self.wrapped_env, key)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    @property
    def base_env(self):
        if isinstance(self.wrapped_env, Wrapper):
            return self.wrapped_env.base_env
        else:
            return self.wrapped_env

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    @property
    def observation_space(self):
        return self.wrapped_env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self, **kwargs):
        return self.wrapped_env.render(**kwargs)

    def __str__(self):
        return '<{}{}>'.format(type(self).__name__, self.env)

    def __repr__(self):
        return str(self)


class ObsWrapper(Wrapper):
    def __init__(self, env):
        super(ObsWrapper, self).__init__(env)

    def wrap_obs(self, obs, info=None):
        raise NotImplementedError()

    def step(self, action):
        obs, r, done, infos = self.wrapped_env.step(action)
        return self.wrap_obs(obs, info=infos), r, done, infos

    def reset(self, env_info=None):
        if env_info is None:
            env_info = {}
        obs = self.wrapped_env.reset()
        return self.wrap_obs(obs, info=env_info)


class FixedEncodeWrapper(ObsWrapper):
    def __init__(self, env, fixed_encoding):
        #Serializable.quick_init(self, locals())
        super(FixedEncodeWrapper, self).__init__(env)
        self.fixed_encoding = fixed_encoding
        assert isinstance(env.observation_space, Box)
        assert len(env.observation_space.shape) == 1
        assert len(fixed_encoding.shape) == 1

        self.inner_dim = env.observation_space.shape[0]

        low = np.r_[env.observation_space.low, fixed_encoding]
        high = np.r_[env.observation_space.high, fixed_encoding]
        self.__observation_space = Box(low, high)

    def wrap_obs(self, obs, info=None):
        obs = np.r_[obs, self.fixed_encoding]
        return obs

    def unwrap_obs(self, obs, info=None):
        if len(obs.shape) == 1:
            return obs[:self.inner_dim]
        else:
            return obs[:,:self.inner_dim]

    @property
    def observation_space(self):
        return self.__observation_space


class TimeLimitWrapper(Wrapper):
    def __init__(self, env, time_limit=-1):
        super(TimeLimitWrapper, self).__init__(env)
        self._t = 0
        self.time_limit = time_limit

    def reset(self):
        self._t = 0
        return self.wrapped_env.reset()

    def step(self, action):
        obs, r, done, infos = self.wrapped_env.step(action)
        self._t += 1
        if self._t >= self.time_limit:
            done = True
        return obs, r, done, infos

