TEMPLATE = subdirs

SUBDIRS = UGV_Core Voxels \
    Direct_lidar_inertial_odometry \
    FAST-LIO2  \
    nav2_smac_planner\
    MPPI-Generic \
    Vector-Pursuit-Controller
    
UGV_Core.depends += Voxels Direct_lidar_inertial_odometry FAST-LIO2 nav2_smac_planner MPPI-Generic Vector-Pursuit-Controller
UGV_Core.file = UGV_Core/UGV_Core.pro
