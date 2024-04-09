# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

## TODO add diff ik
# from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematicsCfg
from omni.isaac.orbit.objects import RigidObjectCfg
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulatorCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.orbit_envs.isaac_env_cfg import EnvCfg, IsaacEnvCfg, PhysxCfg, SimCfg, ViewerCfg
from omni.isaac.orbit.objects.articulated import ArticulatedObjectCfg

from rialto.envs.general.diff_ik import DifferentialInverseKinematicsCfg
##
# Scene settings
##

@configclass
class SceneCfg(RigidObjectCfg):
    meta_info = RigidObjectCfg.MetaInfoCfg(
        # usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/final_scene2.usdz",#/scratch/marcel/objects/modelartrootsiteconvexfix.usd",#"/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/modelartrootsiteconvexfix.usd",#/scratch/marcel/objects/modelartrootsiteconvexfix.usd",
        usd_path="/data/pulkitag/misc/marcel/digital-twin/policy_learning/assets/objects/shelfandmug.usdz",
        # sites_names=["Site"]
    )
    """Meta-information about the ArticulatedObjectCfgarticulated object."""
    init_state = RigidObjectCfg.InitialStateCfg(
        lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0),
        # dof_pos={
        #     "PrismaticJoint": -0.27,
        # },
        # dof_vel={
        #     "PrismaticJoint": 0.0,
        # }
    )
    rigid_props = RigidObjectCfg.RigidBodyPropertiesCfg()

    collision_props= RigidObjectCfg.CollisionPropertiesCfg()
    """Properties to apply to all collisions in the articulation."""
    # articulation_props = ArticulatedObjectCfg.ArticulationRootPropertiesCfg()
    # """Properties to apply to articulation."""
    distractor_paths = []
    

##
# MDP settings
##


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg:
        """Observations for policy group."""

        # global group settings
        enable_corruption: bool = True
        # observation terms
        # -- joint state
        # arm_dof_pos = {"scale": 1.0}
        # arm_dof_pos_scaled = {"scale": 1.0}
        # arm_dof_vel = {"scale": 0.5, "noise": {"name": "uniform", "min": -0.01, "max": 0.01}}
        # tool_dof_pos_scaled = {"scale": 1.0}
        # -- end effector state
        tool_positions = {"scale": 1.0}
        tool_dof_pos_scaled = {"scale":1.0}
        tool_orientations = {"scale": 1.0}
        # -- object state
        objects_pos = {"scale":1.0}
        objects_rot = {"scale":1.0}
        # objects_rot = {"scale":1.0}
        # -- previous action
        # arm_actions = {"scale": 1.0}
        # tool_actions = {"scale": 1.0}
        # -- scene rotation
        # scene_rot = {"scale":1.0}
        # scene_pos = {"scale":1.0}

    # global observation settings
    return_dict_obs_in_group = True
    """Whether to return observations as dictionary or flattened vector within groups."""
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class GoalMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.05, 0.05, 0.05]  # x,y,z


@configclass
class FrameMarkerCfg:
    """Properties for visualization marker."""

    # usd file to import
    usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd"
    # scale of the asset at import
    scale = [0.1, 0.1, 0.1]  # x,y,z

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    simple_reaching_reward = {"weight":1.0}
    success_reward = {"weight":10.0}
    #penalizing_arm_action_rate_l2 = {"weight": 1e-2}
    # penalizing_tool_action_l2 = {"weight": 1e-2}
    #penalizing_arm_dof_velocity_l2 = {"weight": 1e-5}
    # penalizing_tool_dof_velocity_l2 = {"weight": 1e-5}

    # -- robot-centric
    # reaching_object_position_tanh = {"weight": 2.5, "sigma": 0.1}
    # reaching_object_position_l2 = {"weight":2.5}
    # -- action-centric
    # penalizing_arm_action_rate_l2 = {"weight": 0.01}
    # -- object-centric
    # opening_scene_reward = {"weight": 5}

    # gripper_pose = {"weight":1, "close_thresh":0.03}

    # success_reward = {"weight":3.5, "threshold":0.0}



# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     episode_timeout = True  # reset when episode length ended
#     #object_falling = True  # reset when object falls off the table
#     #is_success = False  # reset when object is lifted


@configclass
class ControlCfg:
    """Processing of MDP actions."""

    # action space
    control_type = "inverse_kinematics"  # "default", "inverse_kinematics"
    # decimation: Number of control action updates @ sim dt per policy dt
    decimation = 20 # 10 # TODO maybe we can change to 16
    # relative_rotation = False

    # # configuration loaded when control_type == "inverse_kinematics"
    inverse_kinematics: DifferentialInverseKinematicsCfg = DifferentialInverseKinematicsCfg(
        command_type="pose_abs", # pose_rel, pose_abs
        ik_method="lstsq",
        ik_params={"dt":1}, #1
        # position_command_scale=(0.1, 0.1, 0.1),
        # rotation_command_scale=(0.1, 0.1, 0.1),
    )

@configclass
class RandomizationCfg:
    """Randomization of scene at reset."""

    @configclass
    class RobotInitialPoseCfg:
        """Randomization of object initial pose."""

        # randomize position
        position_min_bound = [-0.0, -0.0, 0]  # position (x,y,z) # TODO: change to -0.15
        position_max_bound = [0.0, 0.0, 0]  # position (x,y,z)
        orientation_min_bound = 0#-0.1 #-0.52 
        orientation_max_bound = 0#0.1 #0.52  

    @configclass
    class ObjectInitialPoseCfg:
        """Randomization of object initial pose."""

        # randomize position
        position_min_bound = [-0.4, -0.1, -0.1]  # position (x,y,z) # TODO: change to -0.15
        position_max_bound = [0.2, 0.3, 0.1]  # position (x,y,z)
        orientation_min_bound = -0.32 #-0.52 
        orientation_max_bound = 0.32#0.52  

    # initialize
    robot_initial_pose: RobotInitialPoseCfg = RobotInitialPoseCfg()
    randomize_pos = True
    randomize_rot = True
    floor_height = -0.5
    randomize_object_name = "Object_268"
    object_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()
    distractor_initial_pose: ObjectInitialPoseCfg = ObjectInitialPoseCfg()

##
# Environment configuration
##


@configclass
class GeneralEnvCfg(IsaacEnvCfg):
    """Configuration for the Lift environment."""

    # General Settings
    env: EnvCfg = EnvCfg(num_envs=4096, env_spacing=6, episode_length_s=1000.0) #5.0
    viewer: ViewerCfg = ViewerCfg(debug_vis=True, eye=(7.5, 7.5, 7.5), lookat=(0.0, 0.0, 0.0))
    # Physics settings
    sim: SimCfg = SimCfg(
        dt=0.005, #0.01
        substeps=1, #1
        physx=PhysxCfg(
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=1024 * 1024,
            # friction_correlation_distance=0.00625,
            # friction_offset_threshold=0.01,
            bounce_threshold_velocity=0.2,
        ),
    )

    reward_type = "drawer_new"
    dense_reward = False

    # Scene Settings
    # -- robot
    # robot: SingleArmManipulatorCfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    # robot.init_state.pos =(0,0,0)

    # -- object
    scene: SceneCfg = SceneCfg()
    # -- table
    # -- visualization marker
    goal_marker: GoalMarkerCfg = GoalMarkerCfg()
    frame_marker: FrameMarkerCfg = FrameMarkerCfg()

    randomization: RandomizationCfg = RandomizationCfg()

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    # terminations: TerminationsCfg = TerminationsCfg()

    # Controller settings
    control: ControlCfg = ControlCfg()
