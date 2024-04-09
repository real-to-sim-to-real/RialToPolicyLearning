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

"""Utility functions relating to math."""

import logging
from typing import Sequence

import numpy as np


def average_quaternions(quaternions: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the average of the given quaternions.

    Args:
        quaternions: A list of quaternions to average.

    Returns:
        The averaged quaternion.
    """
    # Implements the algorithm from:
    # Markley, F. L., Cheng, Y., Crassidis, J. L., & Oshman, Y. (2007).
    # Averaging quaternions. Journal of Guidance, Control, and Dynamics,
    # 30(4), 1193-1197.
    n_quat = len(quaternions)
    assert n_quat > 0, 'Must provide at least one quaternion.'
    weight = 1.0 / n_quat  # Uniform weighting for all quaternions.
    q_matrix = np.vstack(quaternions)
    assert q_matrix.shape == (n_quat, 4)
    m_matrix = np.matmul(weight * np.transpose(q_matrix), q_matrix)
    _, eig_vecs = np.linalg.eigh(m_matrix)
    # The final eigenvector corresponds to the largest eigenvalue.
    return eig_vecs[:, -1]


def calculate_cosine(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Calculates the cosine angle between two vectors.

    This computes cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))

    Args:
        vec1: The first vector. This can have a batch dimension.
        vec2: The second vector. This can have a batch dimension.

    Returns:
        The cosine angle between the two vectors, with the same batch dimension
        as the given vectors.
    """
    if np.shape(vec1) != np.shape(vec2):
        raise ValueError('{} must have the same shape as {}'.format(vec1, vec2))
    ndim = np.ndim(vec1)
    if ndim < 1 or ndim > 2:
        raise ValueError('{} must be 1 or 2 dimensions'.format(vec1))
    axis = 1 if ndim == 2 else 0
    norm_product = (
        np.linalg.norm(vec1, axis=axis) * np.linalg.norm(vec2, axis=axis))
    zero_norms = norm_product == 0
    if np.any(zero_norms):
        logging.warning(
            '%s or %s is all 0s; this may be normal during initialization.',
            str(vec1), str(vec2))
        if ndim == 2:
            norm_product[zero_norms] = 1
        else:
            norm_product = 1
    # Return the batched dot product.
    return np.einsum('...i,...i', vec1, vec2) / norm_product
