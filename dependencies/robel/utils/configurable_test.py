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

"""Tests for configurable."""

import pickle
import tempfile

from absl.testing import absltest

from robel.utils.configurable import configurable

TEST_CONFIGS = {}


@configurable(config_cache=TEST_CONFIGS)
class DummyWithConfig(object):
    def __init__(self, a=1, b=2, c=3):
        self.a = a
        self.b = b
        self.c = c


class ChildDummyWithConfig(DummyWithConfig):
    pass


@configurable(pickleable=True, config_cache=TEST_CONFIGS)
class DummyWithConfigPickleable(object):
    def __init__(self, a=1, b=2, c=3):
        self.a = a
        self.b = b
        self.c = c


class TestConfigurable(absltest.TestCase):
    """Unit tests for configurable."""

    def setUp(self):
        TEST_CONFIGS.clear()

    def test_instance(self):
        """Tests default values."""
        d = DummyWithConfig()
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)
        self.assertEqual(d.c, 3)

    def test_set_config(self):
        """Tests setting a config values."""
        TEST_CONFIGS[DummyWithConfig] = {'a': 4, 'c': 5}

        d = DummyWithConfig()
        self.assertEqual(d.a, 4)
        self.assertEqual(d.b, 2)
        self.assertEqual(d.c, 5)

    def test_set_config_kwargs(self):
        """Tests overriding a config with kwargs."""
        TEST_CONFIGS[DummyWithConfig] = {'a': 4, 'c': 5}

        d = DummyWithConfig(a=7)
        self.assertEqual(d.a, 7)
        self.assertEqual(d.b, 2)
        self.assertEqual(d.c, 5)

    def test_set_config_inheritance(self):
        """Tests config values for a child class."""
        TEST_CONFIGS[ChildDummyWithConfig] = {'a': 4, 'c': 5}

        d1 = ChildDummyWithConfig()
        self.assertEqual(d1.a, 4)
        self.assertEqual(d1.b, 2)
        self.assertEqual(d1.c, 5)

        d2 = DummyWithConfig()
        self.assertEqual(d2.a, 1)
        self.assertEqual(d2.b, 2)
        self.assertEqual(d2.c, 3)

    def test_pickle(self):
        """Tests loading from a pickled object."""
        TEST_CONFIGS[DummyWithConfigPickleable] = {'a': 4, 'c': 5}

        d = DummyWithConfigPickleable(b=8)
        TEST_CONFIGS.clear()

        with tempfile.TemporaryFile() as f:
            pickle.dump(d, f)
            f.seek(0)
            d2 = pickle.load(f)

        self.assertEqual(d2.a, 4)
        self.assertEqual(d2.b, 8)
        self.assertEqual(d2.c, 5)

    def test_pickle_override(self):
        """Tests overriding serialized parameters."""
        TEST_CONFIGS[DummyWithConfigPickleable] = {'a': 4, 'c': 5}

        d = DummyWithConfigPickleable(c=1)
        self.assertEqual(d.a, 4)
        self.assertEqual(d.b, 2)
        self.assertEqual(d.c, 1)

        with tempfile.TemporaryFile() as f:
            pickle.dump(d, f)
            f.seek(0)

            TEST_CONFIGS[DummyWithConfigPickleable] = {'b': 5}

            d2 = pickle.load(f)

        self.assertEqual(d2.a, 4)
        self.assertEqual(d2.b, 5)
        self.assertEqual(d2.c, 1)


if __name__ == '__main__':
    absltest.main()
