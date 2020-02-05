import copy
import argparse
import numpy as np
import open3d as o3
from probreg import cpd
from probreg import l2dist_regs
from probreg import gmmtree
from probreg import filterreg
from probreg import features
import helper

def register_cpd(source, target):
    tf_param, _, _ = cpd.registration_cpd(source, target, update_scale=False)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    return result, tf_param

def register_svr(source, target):
    tf_param = l2dist_regs.registration_svr(source, target)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    return result, tf_param

def register_gmmtree(source, target):
    # tf_param, _ = gmmtree.registration_gmmtree(source, target, maxiter=1000, tree_level=6)
    tf_param, _ = gmmtree.registration_gmmtree(source, target)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    return result, tf_param

def register_filterreg(source, target, objective_type='pt2pt'):
    tf_param, _, _ = filterreg.registration_filterreg(source, target,
            objective_type=objective_type,
            sigma2=None)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    return result, tf_param

def register_filterreg_feature(source, target, objective_type='pt2pt'):
    tf_param, _, _ = filterreg.registration_filterreg(source, target,
            objective_type=objective_type,
            sigma2=1000, feature_fn=features.FPFH())
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)
    return result, tf_param

def register_icp(source, target, voxel_size, objective_type='pt2pt'):
    distance_threshold = voxel_size * 0.1
    trans_init = np.identity(4)
    if objective_type == 'pt2pt':
        tf_param = o3.registration.registration_icp(
                source, target, distance_threshold, trans_init,
                o3.registration.TransformationEstimationPointToPoint())
    elif objective_type == 'pt2pl':
        tf_param = o3.registration.registration_icp(
                source, target, distance_threshold, trans_init,
                o3.registration.TransformationEstimationPointToPlane())
    result = copy.deepcopy(source)
    helper.transform_model(result, tf_param.transformation)
    return result, tf_param


def fast_global_registration(source, target, voxel_size):
    if voxel_size is not None:
        source_local = source.voxel_down_sample(voxel_size)
        target_local = target.voxel_down_sample(voxel_size)
    else:
        source_local = source
        target_local = target

    normal_radius = voxel_size * 2
    # if not source_local.has_normals():
    #     print('test')
    #     helper.estimate_normals_radius(source_local, normal_radius)
    # if not target_local.has_normals():
    #     print('test!')
    #     helper.estimate_normals_radius(target_local, normal_radius, cam_pos=[0, 0, 0])
    # helper.estimate_normals_radius(source_local, normal_radius)
    helper.estimate_normals_radius(target_local, normal_radius, cam_pos=[0, 0, 0])
    source_fpfh = helper.compute_fpfh_feature(source_local, voxel_size)
    target_fpfh = helper.compute_fpfh_feature(target_local, voxel_size)
    distance_threshold = voxel_size * 0.1
    print(":: Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = o3.registration.registration_fast_based_on_feature_matching(source_local, target_local, source_fpfh, target_fpfh, o3.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))
    helper.transform_model(source_local, result.transformation)
    helper.draw_registration_result(source_local, target_local)
    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test different point cloud registration algorithms.')
    parser.add_argument('--method', default='cpd', choices=['cpd', 'svr', 'gmmtree', 'filterreg', 'filterreg_feature', 'icp'], help='Method to be used for point cloud registration.')
    parser.add_argument('--voxel_size', type=float, default=5, help='Voxel size of downsampled point clouds. A value <= 0.0 indicates that no downsampling is used.')
    parser.add_argument('--source_file', help='File containing source model, if --source_is_mesh argument is not set, it is assumed to be a mesh.')
    parser.add_argument('--source_is_mesh', type=bool, default=True, choices=[True, False], help='If false, source model is loaded as point cloud, otherwise as mesh. Default is True.')
    parser.add_argument('--target_file', help='File containing target model, if --target_is_mesh argument is not set, it is assumed to be a pointcloud.')
    parser.add_argument('--target_is_mesh', type=bool, default=False, choices=[True, False], help='If false, traget model is loaded as point cloud, otherwise as mesh. Default is False.')
    parser.add_argument('--initial_transformation', default='none', choices=['none', 'center', 'fast_global_registration'], help='Transformation applied to source model before running registration algorithm.')
    parser.add_argument('--voxel_size_global_reg', type=float, default=None, help='Voxel size of downsampled point clouds for fast global registration. A value <= 0.0 indicates that no downsampling is used.')

    command_line_args = parser.parse_args()

    if command_line_args.source_file is None:
        source_file = '/media/Data/Nextcloud/University/Master/Masterthesis/Environment/ObjectsToTrack/StandfordBunny/bunny-flatfoot_rotated.stl'
        source_is_mesh = True
    else:
        source_file = command_line_args.source_file
        source_is_mesh = command_line_args.source_is_mesh
    if command_line_args.target_file is None:
        target_file = '/media/Data/Masterthesis/Insta360ProVideos/EnvironmentTwoUniPCBoxesObjectTracking/PointClouds/EnvironmentTwoUniPCBoxesObjectTracking_Lens23_Elas_CropedToBunny.ply'
        target_is_mesh = False
    else:
        target_file = command_line_args.target_file
        target_is_mesh = command_line_args.target_is_mesh

    source_original, target_original = helper.load_models(source_filename=source_file, source_is_mesh=source_is_mesh, target_filename=target_file, target_is_mesh=target_is_mesh)

    downsampled = False
    if command_line_args.voxel_size > 0.0:
        voxel_size = command_line_args.voxel_size
        print(':: Downsample source with a voxel size %.3f.' % voxel_size)
        source = source_original.voxel_down_sample(voxel_size)
        print(':: Downsample target with a voxel size %.3f.' % voxel_size)
        target = target_original.voxel_down_sample(voxel_size)
        downsampled = True
        # helper.estimate_normals_radius(source, voxel_size * 2)
        # helper.estimate_normals(source_original)
        helper.estimate_normals_radius(target, voxel_size * 2, [0, 0, 0])
        helper.estimate_normals(target_original, cam_pos=[0, 0, 0])
    else:
        source = source_original
        target = target_original
        voxel_size = 0.05
        downsampled = False
        # helper.estimate_normals(source_original)
        helper.estimate_normals(target, cam_pos=[0, 0, 0])

    voxel_size_global_reg = voxel_size
    if command_line_args.voxel_size_global_reg is not None:
        voxel_size_global_reg = command_line_args.voxel_size_global_reg


    if command_line_args.initial_transformation == 'center':
        print(':: Moving centers of both point clouds in origin so that both be closer to each other.')
        initial_transformation_source = np.identity(4)
        initial_transformation_target = np.identity(4)
        initial_transformation_source[0:3, 3] = (source.get_center() * (-1))
        initial_transformation_target[0:3, 3] = (target.get_center() * (-1))
        helper.transform_model(source, initial_transformation_source)
        helper.transform_model(target, initial_transformation_target)
        if downsampled:
            helper.transform_model(source_original, initial_transformation_source)
            helper.transform_model(target_original, initial_transformation_target)
    elif command_line_args.initial_transformation == 'fast_global_registration':
        print(':: Running fast global registration before applying fine grained registration.')
        result_fast = fast_global_registration(source, target, voxel_size_global_reg)
        initial_transformation_source = result_fast.transformation
        source_old = copy.deepcopy(source)
        helper.transform_model(source, initial_transformation_source)
        if downsampled:
            helper.transform_model(source_original, initial_transformation_source)
        # helper.draw_registration_result(source_old, target, source)

    
    print(':: Ready for registration, showing starting positions of point clouds.')
    helper.draw_registration_result(source, target)
    
    result = None

    if command_line_args.method == 'cpd':
        print(':: Register models with cpd.')
        result, tf_param = register_cpd(source, target)

    elif command_line_args.method == 'svr':
        print(':: Register models with svr.')
        result, tf_param = register_svr(source, target)

    elif command_line_args.method == 'gmmtree':
        print(':: Register models with gmmtree.')
        result, tf_param = register_gmmtree(source, target)
        # print(':: Gmmtree needs a initial alignment, therefore pcd is used first.')
        # source_intermediate, _ = register_cpd(source, target)
        # result, _ = register_gmmtree(source_intermediate, target)
        # helper.draw_registration_result()

    elif command_line_args.method == 'filterreg':
        print(':: Register models with filterreg.')
        result, tf_param = register_filterreg(source, target)

    elif command_line_args.method == 'filterreg_feature':
        print(':: Register models with filterreg_feature.')
        result, tf_param = register_filterreg_feature(source, target)

    elif command_line_args.method == 'icp':
        print(':: Register models with filterreg_feature.')
        result, tf_param = register_icp(source, target, voxel_size)

    if result is not None:
        print(':: Registration successful, results:')
        if command_line_args.method != 'icp':
            print(':: translation: ', tf_param.t)
            print(':: rotation:\n', tf_param.rot)
            print(':: scale: ', tf_param.scale)
        else:
            print(':: transformation: ', tf_param.transformation)
        helper.draw_registration_result(source, target, result)
        if (downsampled):
            result_original = copy.deepcopy(source_original)
            if command_line_args.method != 'icp':
                result_original.points = tf_param.transform(result_original.points)
            else:
                helper.transform_model(result_original, tf_param.transformation)
            helper.draw_registration_result(source_original, target_original, result_original)
