# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment for end-effector pose tracking task for fixed-arm robots."""

from .general_env import GeneralEnv, GeneralEnvCfg

__all__ = ["GeneralEnv", "GeneralEnvCfg"]
