import open3d as o3d
# import numpy as np
import copy

def draw_registration_result(source, target, result=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 1, 0])
    if result is not None:
        result_temp = copy.deepcopy(result)
        result_temp.paint_uniform_color([0, 0, 1])
        o3d.visualization.draw_geometries([source_temp, target_temp, result_temp])
    else:
        o3d.visualization.draw_geometries([source_temp, target_temp])

def mesh_to_point_cloud(mesh):
    pcd = o3d.geometry.PointCloud()
    if mesh.get_geometry_type() == o3d.geometry.Geometry.Type.TriangleMesh:
        pcd.points = mesh.vertices
        pcd.colors = mesh.vertex_colors
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        pcd.normals = mesh.vertex_normals
        return (True, pcd)
    return (False, pcd)

def estimate_normals(pcd, params=None, cam_pos=None):
    if params is None:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(search_param=params)
    if cam_pos is None:
        pcd.orient_normals_to_align_with_direction()
    else:
        pcd.orient_normals_towards_camera_location(cam_pos)

def estimate_normals_radius(pcd, radius, cam_pos=None):
    print(':: Estimate normal with search radius %.3f.' % radius)
    estimate_normals(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30), cam_pos)

def compute_fpfh_feature(pcd, voxel_size):
    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def transform_model(pcd, trans_mat):
    print(':: Transform model by:\n', trans_mat)
    pcd = pcd.transform(trans_mat)

def load_model(filename, is_mesh):
    if is_mesh:
        mesh = o3d.io.read_triangle_mesh(filename)
        (success, model) = mesh_to_point_cloud(mesh)
        if not success:
            print(':: Could not transform mesh to point cloud!')
            exit(-1)
        else:
            print(':: Generated point cloud from mesh.')
    else:
        model = o3d.io.read_point_cloud(filename)
    return model

def load_models(source_filename, target_filename, source_is_mesh=True, target_is_mesh=False):
    print(':: Load source from ', source_filename, '.')
    source = load_model(source_filename, source_is_mesh)
    print(':: Load target from ', target_filename, '.')
    target = load_model(target_filename, target_is_mesh)
    return source, target
