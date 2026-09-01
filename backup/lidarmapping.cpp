#include "lidarmapping.hpp"

LidarMapping::LidarMapping(QObject *parent)
{
}

LidarMapping::~LidarMapping()
{

}

void LidarMapping::h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    laserCloudOri->clear();
    corr_normvect->clear();
    total_residual = 0.0;

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }

    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        qDebug()<<"No Effective Points! \n";
        return;
    }

    res_mean_last = total_residual / effct_feat_num;

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);

        V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
        ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
}

void LidarMapping::publish_path()
{
    //    set_posestamp(msg_body_pose);
    //    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    //    msg_body_pose.header.frame_id = "camera_init";

    //    /*** if path is too large, the rvis will crash ***/
    //    static int jjj = 0;
    //    jjj++;
    //    if (jjj % 10 == 0)
    //    {
    //        path.poses.push_back(msg_body_pose);
    //        pubPath.publish(path);
    //    }
}

void LidarMapping::publish_odometry()
{
    //Возможно локальная СК не поворачивается и находится в системе координат тела body её нужно повернуть этим кватернионном

    pose.x = state_point.pos(0);
    pose.y = state_point.pos(1);
    pose.z = state_point.pos(2);

    vel.x = state_point.vel(0);
    vel.y = state_point.vel(1);
    vel.z = state_point.vel(2);

    quat.x = state_point.rot.coeffs()[0];
    quat.y = state_point.rot.coeffs()[1];
    quat.z = state_point.rot.coeffs()[2];
    quat.w = state_point.rot.coeffs()[3];

    //qDebug()<<"Odometry: x "<<pose.x<<" y "<<pose.y<<" z "<<pose.y<<" Quat "<<quat.w<<"  "<<quat.x<<"  "<<quat.y<<"  "<<quat.z;

//    auto P = kf.get_P();
//    for (int i = 0; i < 6; i ++)
//    {
//        int k = i < 3 ? i + 3 : i - 3;
//        pose_cov.c_pose_covariance[i*6 + 0] = P(k, 3);
//        pose_cov.c_pose_covariance[i*6 + 1] = P(k, 4);
//        pose_cov.c_pose_covariance[i*6 + 2] = P(k, 5);
//        pose_cov.c_pose_covariance[i*6 + 3] = P(k, 0);
//        pose_cov.c_pose_covariance[i*6 + 4] = P(k, 1);
//        pose_cov.c_pose_covariance[i*6 + 5] = P(k, 2);
//    }
}

void LidarMapping::publish_map()
{
    //pcl::toROSMsg(*featsFromMap, laserCloudMap);

     unsigned long long size = featsFromMap->points.size();

    GLfloat * vertices_data; vertices_data = new GLfloat [3 * size];

    GLfloat * color_data; color_data = new GLfloat [3 * size];

    PointType laserCloudWorld;

    COLOR c; uint8_t min = 255, max = 0;

    for (uint32_t i = 0; i < size; i++)
    {
        laserCloudWorld = featsFromMap->points[i];

        vertices_data[i * 3] = 200.0f + laserCloudWorld.y * 100.0f;
        vertices_data[i * 3 + 1] = -50.0f + laserCloudWorld.z  * 100.0f;
        vertices_data[i * 3 + 2] = -200.0f + laserCloudWorld.x  * 100.0f;

        c = GetColor(laserCloudWorld.intensity, min, max);

        //если вдруг захочется покрасить кубик

        color_data[i * 3] = c.r;
        color_data[i * 3 + 1] = c.g;
        color_data[i * 3 + 2] = c.b;
   }

       qDebug()<<" points_map: "<<size;

   if(state == 1) emit DisplayingPointMap (vertices_data, color_data, size, QQuaternion(quat.w, quat.x, quat.y, quat.z));

}

void LidarMapping::publish_effect_world()
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
    }

    //pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
}

void LidarMapping::publish_frame_body()
{
    int size = feats_down_body->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_down_body->points[i], &laserCloudIMUBody->points[i]);
    }


    //pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
}

void LidarMapping::publish_frame_world()
{
        unsigned long long size = feats_undistort->points.size();

        GLfloat * vertices_data; vertices_data = new GLfloat [3 * size];

        GLfloat * color_data; color_data = new GLfloat [3 * size];

        PointType laserCloudWorld;

        //COLOR c; uint8_t min = 255, max = 0;

        for (uint32_t i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld);

            vertices_data[i * 3] = 200.0f + laserCloudWorld.y * 100.0f;
            vertices_data[i * 3 + 1] = -50.0f + laserCloudWorld.z * 100.0f;
            vertices_data[i * 3 + 2] = -200.0f + laserCloudWorld.x * 100.0f;


            //c = GetColor(laserCloudWorld.intensity, min, max);

            //если вдруг захочется покрасить кубик

            color_data[i * 3] = laserCloudWorld.intensity;
            color_data[i * 3 + 1] = laserCloudWorld.intensity;
            color_data[i * 3 + 2] = laserCloudWorld.intensity;
       }

           qDebug()<<" points: "<<size;

       if(state == 1) emit DisplayingPoint(vertices_data, color_data, size, QQuaternion(quat.w, quat.x, quat.y, quat.z));

        //pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (false)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    delete [] vertices_data;
    delete [] color_data;
}

void LidarMapping::map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType mid_point;
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

bool LidarMapping::sync_packages(MeasureGroup &meas)
{
    if (feats_input->empty()) return false;

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {

        *meas.lidar += *feats_input;
        meas.lidar_beg_time = last_timestamp_lidar;
//        meas.lidar = lidar_buffer.front();
//        meas.lidar_beg_time = time_buffer.front();

        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            qDebug()<<"Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }
    if(imu_update && pcl_update >= 3)
    {
     imu_update = false;
     pcl_update = 0;
     new_micro_upd = true;
    }
    else return false;

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_input->header; meas.imu.clear();
    if (imu_time < lidar_end_time) meas.imu.push_back(imu_input);

    lidar_pushed = false;
    return true;
}

void LidarMapping::imu_cbk(const ImuConstPtr msg_in)
{
    mtx_buffer.lock();

    imu_update = true;

    //std::cout<<"IMU got at: "<<msg_in.header<<endl;
   // ImuConstPtr::Ptr msg(new ImuConstPtr(*msg_in));

   // std::shared_ptr<ImuConstPtr> msg = std::make_shared<ImuConstPtr>(msg_in);
   imu_input = std::make_shared<ImuConstPtr>(msg_in);

//    msg->header = msg_in.header - time_diff_lidar_to_imu;// Высчитать всё в секундах!!!!

//    if (abs(timediff_lidar_wrt_imu) > 0.1)
//    {
//        msg->header = timediff_lidar_wrt_imu + msg_in.header;// Высчитать всё в секундах!!!!
//    }

    double timestamp = imu_input->header;

    if (timestamp < last_timestamp_imu)
    {
        qDebug()<<"imu loop back, clear buffer";
        //imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;
    //imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LidarMapping::livox_pcl_cbk(CustomMsg msg)
{
    mtx_buffer.lock();

    NowUpdate = msg.now;

    last_offset_time = msg.points.at(0).offset_time;

    if(first_start) {first_start = false; lastUpdate = NowUpdate; offset_time = last_offset_time; }

    if(imu_update) pcl_update++;

    msg.timebase = offset_time;

    if (msg.header < last_timestamp_lidar)
    {
        qDebug()<<"lidar loop back, clear buffer";
        //lidar_buffer.clear();
    }
    if(new_micro_upd){new_micro_upd = false; last_timestamp_lidar = msg.header;}

    if (abs(last_timestamp_imu - last_timestamp_lidar) > 10.0) //&& !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (!timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1) //&& !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    //PointCloudXYZI::Ptr ptr (new PointCloudXYZI());
    p_pre->process(msg, feats_input);
    //lidar_buffer.push_back(ptr);
    //time_buffer.push_back(last_timestamp_lidar);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void LidarMapping::WindowState(uint8_t state)
{
    this->state = state;
}

void LidarMapping::lasermap_fov_segment()
{
    cub_needrm.clear();
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm);
}

void LidarMapping::points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

void LidarMapping::RGBpointBodyLidarToIMU(const PointType * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LidarMapping::RGBpointBodyToWorld(const PointType * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LidarMapping::pointBodyToWorld(const PointType * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LidarMapping::pointBodyToWorld_ikfom(const PointType * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LidarMapping::dump_lio_state_to_log(FILE *fp)
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a
    fprintf(fp, "\r\n");
    fflush(fp);
}

void LidarMapping::SigHandle(int sig)
{
    flg_exit = true;
    qDebug()<<"catch sig: "<<sig;
}

float LidarMapping::calc_dist(PointType p1, PointType p2)
{
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

COLOR LidarMapping::GetColor(float v, float vmin, float vmax)
{
    COLOR c = {1.0f,1.0f,1.0f}; // white
      float dv;

      if (v < vmin)
         v = vmin;
      if (v > vmax)
         v = vmax;
      dv = vmax - vmin;

      if (v < (vmin + 0.25f * dv)) {
         c.r = 0.0f;
         c.g = 4.0f * (v - vmin) / dv;
      } else if (v < (vmin + 0.5f * dv)) {
         c.r = 0.0;
         c.b = 1.0f + 4.0f * (vmin + 0.25 * dv - v) / dv;
      } else if (v < (vmin + 0.75f * dv)) {
         c.r = 4.0f * (v - vmin - 0.5f * dv) / dv;
         c.b = 0.0f;
      } else {
         c.g = 1.0f + 4.0f * (vmin + 0.75f * dv - v) / dv;
         c.b = 0.0f;
      }

      return(c);
}

void LidarMapping::run()
{  
        NUM_MAX_ITERATIONS = 3;
        map_file_path = "";
        lid_topic = "/livox/lidar";
        imu_topic = "/livox/imu";
        time_diff_lidar_to_imu = 0.0;
        filter_size_corner_min = 0.5;
        filter_size_surf_min = 0.5;
        filter_size_map_min = 0.5;
        cube_len = 1000;
        DET_RANGE = 100.f;
        fov_deg = 360;
        gyr_cov = 0.1;
        acc_cov = 0.1;
        b_gyr_cov = 0.0001;
        b_acc_cov = 0.0001;
        p_pre->blind = 0.5;
        p_pre->lidar_type = AVIA;
        p_pre->N_SCANS = 4;
        p_pre->time_unit = US;
        p_pre->SCAN_RATE = 10;
        p_pre->point_filter_num = 3;
        p_pre->feature_enabled = false;
        pcd_save_interval = -1;
        extrinT = vector<double>{-0.011, -0.02329, 0.04412};
        extrinR = vector<double>{ 1, 0, 0,
                                  0, 1, 0,
                                  0, 0, 1 };

        cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;

        /*** variables definition ***/

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        double epsi[23] = {0.001};
        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        /*** debug record ***/
        FILE *fp;
        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        ofstream fout_pre, fout_out, fout_dbg;
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    //-----------------------------------------------------------------------------------------------------

        while (1)
        {
            if (flg_exit) break;
            if(sync_packages(Measures))
            {
                if (flg_first_scan)
                {
                    first_lidar_time = Measures.lidar_beg_time;
                    p_imu->first_lidar_time = first_lidar_time;
                    flg_first_scan = false;
                    continue;
                }

                p_imu->Process(Measures, kf, feats_undistort_storage);

                state_point = kf.get_x();
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                /******* Publish odometry *******/
                publish_odometry();

                if(NowUpdate - lastUpdate >= 100000000){lastUpdate = NowUpdate; offset_time = last_offset_time;}
                else {*feats_undistort += *feats_undistort_storage; continue;}


                if (feats_undistort->empty() || (feats_undistort == NULL))
                {
                    qDebug()<<"No point, skip this scan!\n";
                    continue;
                }

                flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                                false : true;
                /*** Segment the map in lidar FOV ***/
                lasermap_fov_segment();

                /*** downsample the feature points in a scan ***/
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                feats_down_size = feats_down_body->points.size();
                /*** initialize the map kdtree ***/


                if(ikdtree.Root_Node == nullptr)
                {
                    if(feats_down_size > 5) //???????
                    {
                        ikdtree.set_downsample_param(filter_size_map_min);
                        feats_down_world->resize(feats_down_size);


                        for(int i = 0; i < feats_down_size; i++)
                        {
                            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                        }

                        ikdtree.Build(feats_down_world->points);
                    }
                    continue;
                }

                ikdtree.validnum();

                 //cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<0<<"effect num:"<<effct_feat_num<<endl;

                /*** ICP and iterated Kalman filter update ***/
                if (feats_down_size < 5)
                {
                    qDebug()<<"No point, skip this scan!\n";
                    continue;
                }

                normvec->resize(feats_down_size);
                feats_down_world->resize(feats_down_size);

                V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

                if(1) // If you need to see map point, change to "if(1)"
                {
                    PointVector ().swap(ikdtree.PCL_Storage);
                    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                    featsFromMap->clear();
                    featsFromMap->points = ikdtree.PCL_Storage;
                }

                pointSearchInd_surf.resize(feats_down_size);
                Nearest_Points.resize(feats_down_size);

                /*** iterated state estimation ***/
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV);
                state_point = kf.get_x();
                euler_cur = SO3ToEuler(state_point.rot);
                pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

                /*** add the feature points to map kdtree ***/
                map_incremental();

                /******* Publish points *******/
                publish_path();
                publish_frame_world();
                //publish_frame_body();
                //publish_effect_world();
                publish_map();

                feats_undistort->clear();
            }
        }

        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. pcd save will largely influence the real-time performences **/
        if (pcl_wait_save->size() > 0 && false)
        {
            string file_name = string("scans.pcd");
            string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << file_name<<endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        }

        fout_out.close();
        fout_pre.close();
}










