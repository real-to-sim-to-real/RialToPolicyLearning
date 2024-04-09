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

"""Decorator for passing parameters to environments.

This allows parameterizing environments via `gym.make` for older versions of
gym.
NOTE: From gym 0.12 and onwards, `gym.make` accepts `kwargs`.

Example usage:
>>> robel.set_env_params(
...     'DClawTurnStatic-v0', {'device_path': '/dev/ttyUSB0'})
>>> env = gym.make('DClawTurnStatic-v0')

"""

import importlib
import inspect
import logging
from typing import Any, Dict, Optional, Type

from gym.envs.registration import registry as gym_registry

# Global mapping of environment class to parameters.
_ENV_PARAMS = {}


def set_env_params(env_id: str, params: Dict[str, Any]):
    """Sets the parameters for the given environment ID."""
    if env_id not in gym_registry.env_specs:
        raise ValueError('Unregistered environment ID: {}'.format(env_id))
    spec = gym_registry.env_specs[env_id]
    # Fallback compatibility for older gym versions.
    entry_point = getattr(spec, "entry_point",
                          getattr(spec, "_entry_point", None))
    assert entry_point is not None
    if not callable(entry_point):
        assert isinstance(entry_point, str)
        # Get the class handle of the entry-point string.
        module_path, class_name = entry_point.split(":")
        module = importlib.import_module(module_path)
        entry_point = getattr(module, class_name)

    _ENV_PARAMS[entry_point] = params


def configurable(pickleable: bool = False,
                 config_cache: Optional[Dict[Type, Dict[str, Any]]] = None):
    """Class decorator to allow injection of constructor arguments.

    Example usage:
    >>> @configurable()
    ... class A:
    ...     def __init__(self, b=None, c=2, d='Wow'):
    ...         ...

    >>> set_env_params(A, {'b': 10, 'c': 20})
    >>> a = A()      # b=10, c=20, d='Wow'
    >>> a = A(b=30)  # b=30, c=20, d='Wow'

    TODO(michaelahn): Add interop with gin-config.

    Args:
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the config and constructor arguments.
        config_cache: The dictionary of stored environment parameters to use.
            If not explicitly provided, uses the default global dictionary.
    """
    # pylint: disable=protected-access,invalid-name
    if config_cache is None:
        config_cache = _ENV_PARAMS

    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__

        def __init__(self, *args, **kwargs):
            config = config_cache.get(type(self), {})
            # Allow kwargs to override the config.
            kwargs = {**config, **kwargs}

            logging.debug('Initializing %s with params: %s',
                          type(self).__name__, str(kwargs))

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)

        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments and config.
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'

            def __getstate__(self):
                return {
                    PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }

            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                # Override the saved state with the current config.
                config = config_cache.get(type(self), {})
                kwargs = {**saved_kwargs, **config}

                inst = type(self)(*saved_args, **kwargs)
                self.__dict__.update(inst.__dict__)

            cls.__setstate__ = __setstate__

        return cls

    # pylint: enable=protected-access,invalid-name
    return cls_decorator
