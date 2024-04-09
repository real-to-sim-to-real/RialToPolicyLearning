# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Optional, Tuple

from omni.isaac.orbit.utils import configclass


@configclass
class SceneObjectCfg:
    """Configuration parameters for a robot."""

    @configclass
    class MetaInfoCfg:
        """Meta-information about the manipulator."""

        usd_path: str = MISSING
        """USD file to spawn asset from."""

    @configclass
    class InitialStateCfg:
        """Initial state of the rigid body."""

        # root position
        pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Position of the root in simulation world frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` of the root in simulation world frame.
        Defaults to (1.0, 0.0, 0.0, 0.0).
        """
    
    ##
    # Initialize configurations.
    ##

    meta_info: MetaInfoCfg = MetaInfoCfg()
    """Meta-information about the rigid object."""
    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object."""

