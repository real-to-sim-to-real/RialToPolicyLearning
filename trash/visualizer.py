
obj_prim = xform_prim.GetChildren()[0]
obj_prim = xform_prim.GetChildren()[0]
from pxr import UsdShade
bound_material, _ = UsdShade.MaterialBindingAPI(geom_prim).ComputeBoundMaterial()
mat_prim_str = bound_material.GetPath().pathString
mat_prim = prim_utils.get_prim_at_path(mat_prim_str)
mat_prim.GetChildren()[-1]
diffuse_texture_prim = mat_prim.GetChildren()[-1]
diffuse_texture_prim.GetProperties()
diffuse_texture_prim.GetAttribute('outputs:rgb').Get()
diffuse_texture_prim.GetAttribute('inputs:file').Get()
from omni.isaac.orbit.utils import assets as assets_util
texture_data = assets_util.read_file(diffuse_texture_prim.GetAttribute('inputs:file').Get().resolvedPath)
diffuse_texture_prim.GetAttribute('inputs:file').Get().path
diffuse_texture_prim.GetAttribute('inputs:file').Get().resolvedPath
