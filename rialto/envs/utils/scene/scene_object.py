# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import RigidPrim, RigidPrimView

import omni.isaac.orbit.utils.kit as kit_utils

from .scene_object_cfg import SceneObjectCfg
from .scene_object_data import RigidObjectData, JointData
from omni.isaac.orbit.utils.math import (
    quat_apply,
    quat_mul,
)
from omni.isaac.core.articulations import ArticulationView

class SceneObject:
    """Class for handling rigid objects.

    Rigid objects are spawned from USD files and are encapsulated by a single root prim.
    The root prim is used to apply physics material to the rigid body.

    This class wraps around :class:`RigidPrimView` class from Isaac Sim to support the following:

    * Configuring using a single dataclass (struct).
    * Applying physics material to the rigid body.
    * Handling different rigid body views.
    * Storing data related to the rigid object.

    """

    cfg: SceneObjectCfg
    """Configuration class for the rigid object."""
    # objects: Dict[RigidPrimView]
    """Rigid prim view for the rigid object."""

    def __init__(self, cfg: SceneObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg (RigidObjectCfg): An instance of the configuration class.
        """
        # store inputs
        self.cfg = cfg
        # container for data access
        self._data = {}
        self.data_objects = []
        # buffer variables (filled during spawn and initialize)
        self._spawn_prim_path: str = None

    """
    Properties
    """

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        for i in self.objects:
            return self.objects[i].count

    @property
    def device(self) -> str:
        """Memory device for computation."""
        for i in self.objects:
            return self.objects[i]._device

    @property
    def data(self) -> dict:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        """Spawn a rigid object into the stage (loaded from its USD file).

        Note:
            If inputs `translation` or `orientation` are not :obj:`None`, then they override the initial root
            state specified through the configuration class at spawning.

        Args:
            prim_path (str): The prim path for spawning object at.
            translation (Sequence[float], optional): The local position of prim from its parent. Defaults to None.
            orientation (Sequence[float], optional): The local rotation (as quaternion `(w, x, y, z)`
                of the prim from its parent. Defaults to None.
        """
        # use default arguments
        if translation is None:
            translation = self.cfg.init_state.pos
        if orientation is None:
            orientation = self.cfg.init_state.rot

        # -- save prim path for later
        self._spawn_prim_path = prim_path
        # -- spawn asset if it doesn't exist.
        if not prim_utils.is_prim_path_valid(prim_path):
            # add prim as reference to stage
            print(translation, orientation)
            prim_utils.create_prim(
                self._spawn_prim_path ,
                usd_path=self.cfg.meta_info.usd_path,
                # translation=translation,
                # orientation=orientation,
                # scale=self.cfg.meta_info.scale,
            )
            self.distractor_names = []
            for path in self.cfg.distractor_paths:
                name = path.split("/")[-1].split(".")[0]
                self.distractor_names.append(name)
                prim_utils.create_prim(
                    self._spawn_prim_path+f"/{name}",
                    usd_path=path,
                    # translation=translation,
                    # orientation=orientation,
                    # scale=self.cfg.meta_info.scale,
                )
        else:
            carb.log_warn(f"A prim already exists at prim path: '{prim_path}'. Skipping...")


        # for each body aap
        # apply rigid body properties API
        # RigidPrim(prim_path=prim_path)
        # # -- set rigid body properties
        # kit_utils.set_nested_rigid_body_properties(prim_path, **self.cfg.rigid_props.to_dict())
        # # apply collision properties
        # kit_utils.set_nested_collision_properties(prim_path, **self.cfg.collision_props.to_dict())
        # # create physics material
        # if self.cfg.physics_material is not None:
        #     # -- resolve material path
        #     material_path = self.cfg.physics_material.prim_path
        #     if not material_path.startswith("/"):
        #         material_path = prim_path + "/" + prim_path
        #     # -- create physics material
        #     material = PhysicsMaterial(
        #         prim_path=material_path,
        #         static_friction=self.cfg.physics_material.static_friction,
        #         dynamic_friction=self.cfg.physics_material.dynamic_friction,
        #         restitution=self.cfg.physics_material.restitution,
        #     )
        #     # -- apply physics material
        #     kit_utils.apply_nested_physics_material(prim_path, material.prim_path)

    def initialize(self, prim_paths_expr: Optional[str], envs_positions):
        """Initializes the PhysX handles and internal buffers.

        Note:
            PhysX handles are only enabled once the simulator starts playing. Hence, this function needs to be
            called whenever the simulator "plays" from a "stop" state.

        Args:
            prim_paths_expr (Optional[str], optional): The prim path expression for the prims. Defaults to None.

        Raises:
            RuntimeError: When input `prim_paths_expr` is :obj:`None`, the method defaults to using the last
                prim path set when calling the :meth:`spawn()` function. In case, the object was not spawned
                and no valid `prim_paths_expr` is provided, the function throws an error.
        """
        # default prim path if not cloned
        if prim_paths_expr is None:
            if self._is_spawned is not None:
                self._prim_paths_expr = self._spawn_prim_path
            else:
                raise RuntimeError(
                    "Initialize the object failed! Please provide a valid argument for `prim_paths_expr`."
                )
        else:
            self._prim_paths_expr = prim_paths_expr
        # create handles
        # -- object views
        # Generate one rigidprimview per object
        # TODO:
        self.objects = {}
        self.joints = {}
        prim = prim_utils.get_prim_at_path('/World/envs/env_0/Scene')
        prim_children = prim_utils.get_prim_children(prim)
        for child in prim_children:
            child = prim_utils.get_prim_children(child)[0]

            path = prim_utils.get_prim_path(child)
            name = path.split("/")[-1]
            print(name)
            print(prim_utils.get_prim_object_type(path))
            if prim_utils.get_prim_object_type(path) == "rigid_body":
                print(path)
                # TODO: path here should be with a * in env_0
                path = path.replace("env_0","*")
                print(path)
                self.objects[name] = RigidPrimView(path, reset_xform_properties=False)
                object = self.objects[name]
                self._data[name] = RigidObjectData()
                self.data_objects.append(name)

                object.initialize()
                # set the default state
                object.post_reset()
                # from pxr import PhysicsSchemaTools
                # from omni.physx import get_physx_interface, get_physx_scene_query_interface
                # scene = get_physx_scene_query_interface()
                # path = scene.encodeSdfPath(prim.GetPath())
                # scene.overlap_mesh_any(path[0], path[1])
                # Get joints in body
                grandsons = prim_utils.get_prim_children(child)
                for grandson in grandsons:
                    grandson_path = prim_utils.get_prim_path(grandson)
                    grandson_name = grandson_path.split("/")[-1]
                    if grandson_name == "FixedJoint":
                        print("Next, this is a fixed joint", grandson_name)
                        continue

                    grandson_name = name+"/"+grandson_name
                    print(grandson_name)
                    print(prim_utils.get_prim_object_type(grandson_path))

                    if prim_utils.get_prim_object_type(grandson_path) == "joint":
                        self.joints[grandson_name] = ArticulationView(path, reset_xform_properties=False)
                        joint = self.joints[grandson_name]
                        self._data[grandson_name] = JointData()
                        self.data_objects.append(grandson_name)

                        joint.initialize()
                        # set the default state
                        joint.post_reset()

            elif prim_utils.get_prim_object_type(path) == "xform": # This means its a distractor object
                if not "Object_282" in path:
                    rigid_body = prim_utils.get_prim_children(child)[0]
                    path = prim_utils.get_prim_path(rigid_body)
                    name = path.split("/")[-3]
                    path = path.replace("env_0","*")

                    self.objects[name] = RigidPrimView(path, reset_xform_properties=False)
                    object = self.objects[name]
                    self._data[name] = RigidObjectData()

                    object.initialize()
                    # set the default state
                    object.post_reset()




        # set properties over all instances
        # -- meta-information
        self._process_info_cfg(envs_positions)
        # create buffers
        self._create_buffers()
        self.object_coms = {}
        self.joints_defaults = {}
        for object_name in self.objects:
            self.object_coms[object_name] = (self.objects[object_name].get_coms()[0][0], self.objects[object_name].get_coms()[1][0]) 
        # self.add_distractors(prim_paths_expr, envs_positions)

    # def add_distractors(self, prim_paths, env_pos):

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        """Resets all internal buffers.

        Args:
            env_ids (Optional[Sequence[int]], optional): The indices of the object to reset.
                Defaults to None (all instances).
        """
        pass

    def update_buffers(self, dt: float = None):
        """Update the internal buffers.

        The time step ``dt`` is used to compute numerical derivatives of quantities such as joint
        accelerations which are not provided by the simulator.

        Args:
            dt (float, optional): The amount of time passed from last `update_buffers` call. Defaults to None.
        """
        # frame states
        for object_name  in self.objects:
            object = self.objects[object_name]
            position_w, quat_w = object.get_world_poses(indices=self._ALL_INDICES, clone=False)
            self._data[object_name].root_state_w[:, 0:3] = position_w
            self._data[object_name].root_state_w[:, 3:7] = quat_w
            self._data[object_name].root_state_w[:, 7:] = object.get_velocities(indices=self._ALL_INDICES, clone=False)

        for joint_name in self.joints:
            joint = self.joints[joint_name]
            pos = joint.get_joint_positions(indices=self._ALL_INDICES, clone=False)
            vel = joint.get_joint_velocities(indices=self._ALL_INDICES, clone=False)
            self._data[joint_name].root_state_dof[:, 0] = pos.reshape(-1)
            self._data[joint_name].root_state_dof[:, 1] = vel.reshape(-1)

    """
    Operations - State.
    """

    def set_root_state(self, root_states, env_ids: Optional[Sequence[int]] = None):
        """Sets the root state (pose and velocity) of the actor over selected environment indices.

        Args:
            root_states Dict(torch.Tensor): Input root state for the actor, shape: (len(env_ids), 13).
            env_ids (Optional[Sequence[int]]): Environment indices.
                If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        if env_ids is None:
            env_ids = self._ALL_INDICES
            
        # set into simulation
        for object_name in root_states:
            if "joint" in object_name.lower():
                print("here")
                root_state = root_states[object_name]
                self.joints[object_name].set_joint_positions(root_state[:,0].reshape(-1, 1), indices=env_ids)
                self.joints[object_name].set_joint_velocities(root_state[:,1].reshape(-1,1), indices=env_ids)
                
            else:
                root_state = torch.hstack([root_states[object_name], torch.zeros((len(env_ids), 6)).to(self.device)])
                self.objects[object_name].set_world_poses(root_state[:, 0:3], root_state[:, 3:7], indices=env_ids)
                self.objects[object_name].set_velocities(root_state[:, 7:], indices=env_ids)

                # TODO: Move these to reset_buffers call.
                # note: we need to do this here since tensors are not set into simulation until step.
                # set into internal buffers
                self._data[object_name].root_state_w[env_ids] = root_state.clone()



    def get_default_root_state(self, env_ids: Optional[Sequence[int]] = None, clone=True) -> torch.Tensor:
        """Returns the default/initial root state of actor.

        Args:
            env_ids (Optional[Sequence[int]], optional): Environment indices.
                Defaults to None (all environment indices).
            clone (bool, optional): Whether to return a copy or not. Defaults to True.

        Returns:
            torch.Tensor: The default/initial root state of the actor, shape: (len(env_ids), 13).
        """

        # use ellipses object to skip initial indices.
        if env_ids is None:
            env_ids = ...
        # return copy
        default_states = {}
        for object_name in self._default_root_states:
            if clone:
                default_states[object_name] = torch.clone(self._default_root_states[object_name][env_ids])

            else:
                default_states[object_name] =self._default_root_states[object_name][env_ids]

        return default_states


    """
    Internal helper.
    """
    ## TODO: update this by the initial locations of each object in the scene
    def _process_info_cfg(self, envs_positions) -> None:
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        # default_root_state = (
        #     tuple(self.cfg.init_state.pos)
        #     + tuple(self.cfg.init_state.rot)
        #     + tuple(self.cfg.init_state.lin_vel)
        #     + tuple(self.cfg.init_state.ang_vel)
        # )
        self._default_root_states = {}
        for object_name in self.objects:
            new_pos = quat_apply(torch.Tensor([0.7071, 0.7071, 0, 0]).repeat(self.count, 1).to(self.device), self.objects[object_name].get_world_poses()[0]-envs_positions)
            new_rot = quat_mul(torch.Tensor([0.7071, 0.7071, 0, 0]).repeat(self.count, 1).to(self.device), self.objects[object_name].get_world_poses()[1])
            self._default_root_states[object_name] = torch.hstack([new_pos,new_rot])
        
        for joint_name in self.joints:
            new_pos = self.joints[joint_name].get_joints_default_state().positions
            new_vel = torch.zeros_like(new_pos) 
            self._default_root_states[joint_name] = torch.hstack([new_pos,new_vel])
            
            # self._default_root_states[object_name] = torch.hstack(quat_apply([self.objects[object_name].get_world_poses()[0],torch.Tensor([0.7071, 0.7071, 0, 0]).repeat(self.count, 1).to(self.device)),self.objects[object_name].get_world_poses()[0]])
        # self._default_root_states = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        # self._default_root_states = self._default_root_states.repeat(self.count, 1)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.count, dtype=torch.long, device=self.device)

        # -- frame states
        for object_name in self.objects:
            self._data[object_name].root_state_w = torch.zeros(self.count, 13, dtype=torch.float, device=self.device)
        # -- frame states
        for joint_name in self.joints:
            self._data[joint_name].root_state_dof = torch.zeros(self.count, 2, dtype=torch.float, device=self.device)