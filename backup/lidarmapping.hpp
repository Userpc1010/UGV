#ifndef LIDARMAPPING_H
#define LIDARMAPPING_H

#include <QThread>
#include <QQuaternion>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <so3_math.hpp>
#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include "IMU_Processing.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include "preprocess.hpp"
#include <ikd-Tree/ikd_Tree.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

typedef struct {
    float r = 0.0, g = 0.0, b = 0.0;
} COLOR;

class LidarMapping: public QThread
{
  Q_OBJECT

private:
    /**************************/

    static inline float res_last[100000] = {0.0};
    static inline float DET_RANGE = 300.0f;
    static inline const float MOV_THRESHOLD = 1.5f;
    static inline double time_diff_lidar_to_imu = 0.0;

    mutex mtx_buffer;
    condition_variable sig_buffer;

    static inline string root_dir = ROOT_DIR;
    string map_file_path, lid_topic, imu_topic;

    static inline double res_mean_last = 0.05, total_residual = 0.0;
    static inline double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
    static inline double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
    static inline double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
    static inline double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
    static inline int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
    static inline int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
    static inline bool   point_selected_surf[100000] = {0};
    static inline bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;

    vector<vector<int>>  pointSearchInd_surf;
    vector<BoxPointType> cub_needrm;
    static inline vector<PointVector>  Nearest_Points;
    vector<double>       extrinT = {0.0, 0.0, 0.0};
    vector<double>       extrinR = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    uint64_t lastUpdate = 0;
    uint64_t NowUpdate = 0;
    uint64_t last_offset_time = 0;
    uint64_t offset_time = 0;

    bool first_start = true;

    bool imu_update = false;
    bool new_micro_upd = false;
    uint8_t pcl_update = 0;

//    deque<double>                     time_buffer;
//    deque<PointCloudXYZI::Ptr>        lidar_buffer;
//    deque<std::shared_ptr<ImuConstPtr>> imu_buffer;

    std::shared_ptr<ImuConstPtr> imu_input;

    static inline pcl::PointCloud<PointType>::Ptr featsFromMap{ new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr feats_undistort{ new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr feats_undistort_storage{ new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr feats_input { new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr feats_down_body{ new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr feats_down_world{ new pcl::PointCloud<PointType>()};
    static inline pcl::PointCloud<PointType>::Ptr normvec{new pcl::PointCloud<PointType>(100000, 1)};
    static inline pcl::PointCloud<PointType>::Ptr laserCloudOri{ new pcl::PointCloud<PointType>(100000, 1)};
    static inline pcl::PointCloud<PointType>::Ptr corr_normvect{  new pcl::PointCloud<PointType>(100000, 1)};
    PointCloudXYZI::Ptr _featsArray;

    static inline pcl::VoxelGrid<PointType> downSizeFilterSurf;
    static inline pcl::VoxelGrid<PointType> downSizeFilterMap;

    static inline KD_TREE<PointType> ikdtree;

    Vector3f XAxisPoint_body = Vector3f(LIDAR_SP_LEN, 0.0, 0.0);
    Vector3f XAxisPoint_world = Vector3f(LIDAR_SP_LEN, 0.0, 0.0);
    Vector3d euler_cur;
    Vector3d position_last = {0.0, 0.0, 0.0};
    Vector3d Lidar_T_wrt_IMU = {0.0, 0.0, 0.0};
    Matrix3d Lidar_R_wrt_IMU = Matrix3d::Identity();

    /*** EKF inputs and output ***/
    MeasureGroup Measures;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    state_ikfom state_point;
    vect3 pos_lid;

    //nav_msgs::Path path;
    //nav_msgs::Odometry odomAftMapped;
    //geometry_msgs::Quaternion geoQuat;
    //geometry_msgs::PoseStamped msg_body_pose;
    Quat_f quat;
    Pose pose;
    Velocity vel;
    Pose_cov pose_cov;

    shared_ptr<Preprocess> p_pre = std::make_shared<Preprocess>();
    shared_ptr<ImuProcess> p_imu = std::make_shared<ImuProcess>();

    static inline pcl::PointCloud<PointType> * pcl_wait_pub = new pcl::PointCloud<PointType>(500000, 1);
    static inline pcl::PointCloud<PointType> * pcl_wait_save = new pcl::PointCloud<PointType>();

    int process_increments = 0;
    double lidar_mean_scantime = 0.0;
    int    scan_num = 0;

    double timediff_lidar_wrt_imu = 0.0;
    bool   timediff_set_flg = false;

    BoxPointType LocalMap_Points;
    bool Localmap_Initialized = false;

    bool publish_update = false;

    uint8_t state = 0;

protected:

    void run() override;

public:
    LidarMapping(QObject *parent = nullptr);
    ~LidarMapping();


    static void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
    void publish_path();
    void publish_odometry();
    void publish_map();
    void publish_effect_world();
    void publish_frame_body();
    void publish_frame_world();
    void map_incremental();
    bool sync_packages(MeasureGroup &meas);

    void lasermap_fov_segment();
    void points_cache_collect();
    void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
    void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
    template<typename T>void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po){
        V3D p_body(pi[0], pi[1], pi[2]);
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

        po[0] = p_global(0);
        po[1] = p_global(1);
        po[2] = p_global(2);
    }
    void pointBodyToWorld(PointType const * const pi, PointType * const po);
    void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s);
    inline void dump_lio_state_to_log(FILE *fp);
    void SigHandle(int sig);
    float calc_dist(PointType p1, PointType p2);
    COLOR GetColor(float v,float vmin,float vmax);

public slots:

    void imu_cbk(const ImuConstPtr msg_in);
    void livox_pcl_cbk(CustomMsg msg);
    void WindowState (uint8_t state);

signals:

void DisplayingPoint(GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation);
void DisplayingPointMap(GLfloat* vertices_buffer, GLfloat* color_buffer, unsigned long long counter, QQuaternion rotation);

};

#endif // LIDARMAPPING_H
