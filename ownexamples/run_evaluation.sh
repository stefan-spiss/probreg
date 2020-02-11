#!/bin/bash

echo "**************************** CPD ***************************"
python registrationExample.py --method=cpd --initial_transformation none --voxel_size 10
python registrationExample.py --method=cpd --initial_transformation none --voxel_size 5
python registrationExample.py --method=cpd --initial_transformation none --voxel_size 1
python registrationExample.py --method=cpd --initial_transformation center --voxel_size 10
python registrationExample.py --method=cpd --initial_transformation center --voxel_size 5
python registrationExample.py --method=cpd --initial_transformation center --voxel_size 1
python registrationExample.py --method=cpd --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=cpd --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=cpd --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** GMMReg ***************************"
python registrationExample.py --method=gmmreg --initial_transformation none --voxel_size 10
python registrationExample.py --method=gmmreg --initial_transformation none --voxel_size 5
python registrationExample.py --method=gmmreg --initial_transformation none --voxel_size 1
python registrationExample.py --method=gmmreg --initial_transformation center --voxel_size 10
python registrationExample.py --method=gmmreg --initial_transformation center --voxel_size 5
python registrationExample.py --method=gmmreg --initial_transformation center --voxel_size 1
python registrationExample.py --method=gmmreg --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=gmmreg --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=gmmreg --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** SVR ***************************"
python registrationExample.py --method=svr --initial_transformation none --voxel_size 10
python registrationExample.py --method=svr --initial_transformation none --voxel_size 5
python registrationExample.py --method=svr --initial_transformation none --voxel_size 1
python registrationExample.py --method=svr --initial_transformation center --voxel_size 10
python registrationExample.py --method=svr --initial_transformation center --voxel_size 5
python registrationExample.py --method=svr --initial_transformation center --voxel_size 1
python registrationExample.py --method=svr --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=svr --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=svr --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** GMMTree ***************************"
python registrationExample.py --method=gmmtree --initial_transformation none --voxel_size 10
python registrationExample.py --method=gmmtree --initial_transformation none --voxel_size 5
python registrationExample.py --method=gmmtree --initial_transformation none --voxel_size 1
python registrationExample.py --method=gmmtree --initial_transformation center --voxel_size 10
python registrationExample.py --method=gmmtree --initial_transformation center --voxel_size 5
python registrationExample.py --method=gmmtree --initial_transformation center --voxel_size 1
python registrationExample.py --method=gmmtree --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=gmmtree --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=gmmtree --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** FilterReg ***************************"
python registrationExample.py --method=filterreg --initial_transformation none --voxel_size 10
python registrationExample.py --method=filterreg --initial_transformation none --voxel_size 5
python registrationExample.py --method=filterreg --initial_transformation none --voxel_size 1
python registrationExample.py --method=filterreg --initial_transformation center --voxel_size 10
python registrationExample.py --method=filterreg --initial_transformation center --voxel_size 5
python registrationExample.py --method=filterreg --initial_transformation center --voxel_size 1
python registrationExample.py --method=filterreg --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=filterreg --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=filterreg --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** FilterReg Features ***************************"
python registrationExample.py --method=filterreg_feature --initial_transformation none --voxel_size 10
python registrationExample.py --method=filterreg_feature --initial_transformation none --voxel_size 5
python registrationExample.py --method=filterreg_feature --initial_transformation none --voxel_size 1
python registrationExample.py --method=filterreg_feature --initial_transformation center --voxel_size 10
python registrationExample.py --method=filterreg_feature --initial_transformation center --voxel_size 5
python registrationExample.py --method=filterreg_feature --initial_transformation center --voxel_size 1
python registrationExample.py --method=filterreg_feature --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=filterreg_feature --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=filterreg_feature --initial_transformation fast_global_registration --voxel_size 1

echo "**************************** ICP ***************************"
python registrationExample.py --method=icp --initial_transformation none --voxel_size 10
python registrationExample.py --method=icp --initial_transformation none --voxel_size 5
python registrationExample.py --method=icp --initial_transformation none --voxel_size 1
python registrationExample.py --method=icp --initial_transformation center --voxel_size 10
python registrationExample.py --method=icp --initial_transformation center --voxel_size 5
python registrationExample.py --method=icp --initial_transformation center --voxel_size 1
python registrationExample.py --method=icp --initial_transformation fast_global_registration --voxel_size 10
python registrationExample.py --method=icp --initial_transformation fast_global_registration --voxel_size 5
python registrationExample.py --method=icp --initial_transformation fast_global_registration --voxel_size 1
