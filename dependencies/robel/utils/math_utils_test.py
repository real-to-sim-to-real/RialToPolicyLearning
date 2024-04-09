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

"""Tests for math_utils."""

from absl.testing import absltest
import numpy as np
from transforms3d.euler import euler2quat, quat2euler

from robel.utils.math_utils import average_quaternions, calculate_cosine


class AverageQuaternionsTest(absltest.TestCase):
    """Tests for `average_quaternions`."""

    def test_identity(self):
        """Average one quaternion should equal itself."""
        test_quat = euler2quat(np.pi / 4, np.pi / 4, np.pi / 4)
        avg_quat = average_quaternions([test_quat])
        np.testing.assert_array_almost_equal(avg_quat, test_quat)

    def test_multiple_identity(self):
        """Average multiple copies of a quaternion should equal itself."""
        test_quat = euler2quat(np.pi / 4, np.pi / 4, np.pi / 4)
        avg_quat = average_quaternions([test_quat, test_quat, test_quat])
        np.testing.assert_array_almost_equal(avg_quat, test_quat)

    def test_average_two(self):
        """Averaging two different quaternions."""
        quat1 = euler2quat(np.pi / 4, 0, 0)
        quat2 = euler2quat(-np.pi / 4, 0, 0)
        avg_quat = average_quaternions([quat1, quat2])
        result = quat2euler(avg_quat)
        np.testing.assert_array_almost_equal(result, [0, 0, 0])


class CalculateCosineTest(absltest.TestCase):
    """Tests for `calculate_cosine`."""

    def test_identical(self):
        """Two of the same vectors are completely aligned."""
        v1 = np.array([1, 0])
        self.assertAlmostEqual(calculate_cosine(v1, v1), 1)

    def test_parallel(self):
        """Two parallel vectors."""
        v1 = np.array([1, 2])
        v2 = np.array([2, 4])
        self.assertAlmostEqual(calculate_cosine(v1, v2), 1)

    def test_opposite(self):
        """Two parallel vectors."""
        v1 = np.array([1, 2])
        v2 = np.array([-1, -2])
        self.assertAlmostEqual(calculate_cosine(v1, v2), -1)

    def test_orthogonal(self):
        """Two orthogonal vectors."""
        v1 = np.array([1, 1])
        v2 = np.array([1, -1])
        self.assertAlmostEqual(calculate_cosine(v1, v2), 0)

    def test_batched(self):
        """Multiple vectors."""
        v1 = np.array([[1, 1], [2, 2]])
        v2 = np.array([[1, -1], [3, 3]])
        np.testing.assert_array_almost_equal(calculate_cosine(v1, v2), [0, 1])

    def test_zero(self):
        """Tests when the norm is 0."""
        v1 = np.array([1, 0])
        v2 = np.array([0, 0])
        self.assertAlmostEqual(calculate_cosine(v1, v2), 0)

    def test_zero_batched(self):
        """Tests when the norm is 0."""
        v1 = np.array([[1, 0], [1, 1]])
        v2 = np.array([[0, 0], [2, 2]])
        np.testing.assert_array_almost_equal(calculate_cosine(v1, v2), [0, 1])


if __name__ == '__main__':
    absltest.main()
