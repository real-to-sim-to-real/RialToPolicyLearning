import numpy as np
import meshcat
import meshcat.geometry as mcg


def meshcat_pcd_show(mc_vis, point_cloud, color=None, name=None, size=0.001, debug=False):
    """
    Function to show a point cloud using meshcat. 

    mc_vis (meshcat.Visualizer): Interface to the visualizer 
    point_cloud (np.ndarray): Shape Nx3 or 3xN
    color (np.ndarray or list): Shape (3,)
    """
    # color_orig = copy.deepcopy(color)
    if point_cloud.shape[0] != 3:
        point_cloud = point_cloud.swapaxes(0, 1)
    if color is None:
        color = np.zeros_like(point_cloud)
    else:
        # color = int('%02x%02x%02x' % color, 16)
        if not isinstance(color, np.ndarray):
            color = np.asarray(color).astype(np.float32)
        color = np.clip(color, 0, 255)
        color = np.tile(color.reshape(3, 1), (1, point_cloud.shape[1]))
        color = color.astype(np.float32)
    if name is None:
        name = 'scene/pcd'

    if debug:
        print("here in meshcat_pcd_show")
        from IPython import embed; embed()

    mc_vis[name].set_object(
        mcg.Points(
            mcg.PointsGeometry(point_cloud, color=(color / 255)),
            mcg.PointsMaterial(size=size)
    ))


def meshcat_multiple_pcd_show(mc_vis, point_cloud_list, color_list=None, name_list=None, rand_color=False):
    colormap = cm.get_cmap('brg', len(point_cloud_list))
    # colormap = cm.get_cmap('gist_ncar_r', len(point_cloud_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(point_cloud_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(point_cloud_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist())
        else:
            color_list = colors

    if name_list is None:
        name_list = [f'scene/pcd_list_{i}' for i in range(len(point_cloud_list))]
    
    for i, pcd in enumerate(point_cloud_list):
        if pcd.shape[0] > 0:
            meshcat_pcd_show(mc_vis, pcd, color=color_list[i][:3].astype(np.int8).tolist(), name=name_list[i])

    
def meshcat_frame_show(mc_vis, name, transform=None, length=0.1, radius=0.008, opacity=1.):
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    Args:
        mc_vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        transform (np.ndarray): 4 x 4 matrix representing the pose
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array([[length / 2, 0, 0],
    [0, length / 2, 0],
    [0, 0, length / 2]])

    axes_name = ['x', 'y', 'z']
    colors = [0xff0000, 0x00ff00, 0x0000ff]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i in range(3):
        material = meshcat.geometry.MeshLambertMaterial(
        color=colors[i], opacity=opacity)
        mc_vis[name][axes_name[i]].set_object(
        meshcat.geometry.Cylinder(length, radius), material)
        X = meshcat.transformations.rotation_matrix(
        np.pi/2, rotation_axes[i])
        X[0:3, 3] = delta_xyz[i]
        if transform is not None:
            X = np.matmul(transform, X)
        mc_vis[name][axes_name[i]].set_transform(X)


def meshcat_trimesh_show(mc_vis, name, trimesh_mesh, color=(128, 128, 128), opacity=1.0):
    verts = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    if not isinstance(color, tuple):
        color = tuple(color)

    color = int('%02x%02x%02x' % color, 16)

    material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
    mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
    # mc_vis[name].set_object(mcg_mesh)
    mc_vis[name].set_object(mcg_mesh, material)


def trimesh_scene_to_mesh(trimesh_scene):
    import trimesh
    verts = []
    faces = []
    meshes = []
    for i, geom in enumerate(trimesh_scene.geometry.values()):
        verts.append(geom.vertices)
        faces.append(geom.faces)
        meshes.append(geom)

    # trimesh_mesh = trimesh.Trimesh(verts, faces)
    trimesh_mesh = trimesh.util.concatenate(meshes)
    return trimesh_mesh
