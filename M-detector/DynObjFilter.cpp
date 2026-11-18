#include <iostream>
#include <vector>
#include <random>
#include <m-detector/DynObjFilter.h>
// #include <algorithm>
// #include <chrono>
// #include <execution>

void DynObjFilter::init(const MDetectorConfig& cfg)
{
    buffer_delay = cfg.buffer_delay;
    buffer_size = cfg.buffer_size;
    points_num_perframe = cfg.points_num_perframe;
    depth_map_dur = cfg.depth_map_dur;
    max_depth_map_num = cfg.max_depth_map_num;
    max_pixel_points = cfg.max_pixel_points;
    frame_dur  = cfg. frame_dur;
    dataset  = cfg. dataset;
    self_x_f  = cfg. self_x_f;
    self_x_b  = cfg. self_x_b;
    self_y_l  = cfg. self_y_l;
    self_y_r  = cfg. self_y_r;
    blind_dis  = cfg. blind_dis;
    fov_up  = cfg. fov_up;
    fov_down  = cfg. fov_down;
    fov_cut  = cfg. fov_cut;
    fov_left  = cfg. fov_left;
    fov_right  = cfg. fov_right;
    checkneighbor_range  = cfg. checkneighbor_range;
    stop_object_detect  = cfg. stop_object_detect;
    depth_thr1  = cfg. depth_thr1;
    enter_min_thr1  = cfg. enter_min_thr1;
    enter_max_thr1  = cfg. enter_max_thr1;
    map_cons_depth_thr1  = cfg. map_cons_depth_thr1;
    map_cons_hor_thr1  = cfg. map_cons_hor_thr1;
    map_cons_ver_thr1  = cfg. map_cons_ver_thr1;
    map_cons_hor_dis1  = cfg. map_cons_hor_dis1;
    map_cons_ver_dis1  = cfg. map_cons_ver_dis1;
    depth_cons_depth_thr1  = cfg. depth_cons_depth_thr1;
    depth_cons_depth_max_thr1  = cfg. depth_cons_depth_max_thr1;
    depth_cons_hor_thr1  = cfg. depth_cons_hor_thr1;
    depth_cons_ver_thr1  = cfg. depth_cons_ver_thr1;
    enlarge_z_thr1  = cfg. enlarge_z_thr1;
    enlarge_angle  = cfg. enlarge_angle;
    enlarge_depth  = cfg. enlarge_depth;
    occluded_map_thr1  = cfg. occluded_map_thr1;
    case1_interp_en  = cfg. case1_interp_en;
    k_depth_min_thr1  = cfg. k_depth_min_thr1;
    d_depth_min_thr1  = cfg. d_depth_min_thr1;
    k_depth_max_thr1  = cfg. k_depth_max_thr1;
    d_depth_max_thr1  = cfg. d_depth_max_thr1;
    v_min_thr2  = cfg. v_min_thr2;
    acc_thr2  = cfg. acc_thr2;
    map_cons_depth_thr2  = cfg. map_cons_depth_thr2;
    map_cons_hor_thr2  = cfg. map_cons_hor_thr2;
    map_cons_ver_thr2  = cfg. map_cons_ver_thr2;
    occ_depth_thr2  = cfg. occ_depth_thr2;
    occ_hor_thr2  = cfg. occ_hor_thr2;
    occ_ver_thr2  = cfg. occ_ver_thr2;
    depth_cons_depth_thr2  = cfg. depth_cons_depth_thr2;
    depth_cons_depth_max_thr2  = cfg. depth_cons_depth_max_thr2;
    depth_cons_hor_thr2  = cfg. depth_cons_hor_thr2;
    depth_cons_ver_thr2  = cfg. depth_cons_ver_thr2;
    k_depth2  = cfg. k_depth2;
    occluded_times_thr2  = cfg. occluded_times_thr2;
    case2_interp_en  = cfg. case2_interp_en;
    k_depth_max_thr2  = cfg. k_depth_max_thr2;
    d_depth_max_thr2  = cfg. d_depth_max_thr2;
    v_min_thr3  = cfg. v_min_thr3;
    acc_thr3  = cfg. acc_thr3;
    map_cons_depth_thr3  = cfg. map_cons_depth_thr3;
    map_cons_hor_thr3  = cfg. map_cons_hor_thr3;
    map_cons_ver_thr3  = cfg. map_cons_ver_thr3;
    occ_depth_thr3  = cfg. occ_depth_thr3;
    occ_hor_thr3  = cfg. occ_hor_thr3;
    occ_ver_thr3  = cfg. occ_ver_thr3;
    depth_cons_depth_thr3  = cfg. depth_cons_depth_thr3;
    depth_cons_depth_max_thr3  = cfg. depth_cons_depth_max_thr3;
    depth_cons_hor_thr3  = cfg. depth_cons_hor_thr3;
    depth_cons_ver_thr3  = cfg. depth_cons_ver_thr3;
    k_depth3  = cfg. k_depth3;
    occluding_times_thr3  = cfg. occluding_times_thr3;
    case3_interp_en  = cfg. case3_interp_en;
    k_depth_max_thr3  = cfg. k_depth_max_thr3;
    d_depth_max_thr3  = cfg. d_depth_max_thr3;
    interp_hor_thr  = cfg. interp_hor_thr;
    interp_ver_thr  = cfg. interp_ver_thr;
    interp_thr1  = cfg. interp_thr1;
    interp_static_max  = cfg. interp_static_max;
    interp_start_depth1  = cfg. interp_start_depth1;
    interp_kp1  = cfg. interp_kp1;
    interp_kd1  = cfg. interp_kd1;
    interp_thr2  = cfg. interp_thr2;
    interp_thr3  = cfg. interp_thr3;
    dyn_filter_en  = cfg. dyn_filter_en;
    debug_en  = cfg. debug_en;
    laserCloudSteadObj_accu_limit  = cfg. laserCloudSteadObj_accu_limit;
    voxel_filter_size  = cfg. voxel_filter_size;
    cluster_coupled  = cfg. cluster_coupled;
    cluster_future  = cfg. cluster_future;
    Cluster.cluster_extend_pixel  = cfg. cluster_extend_pixel;
    Cluster.cluster_min_pixel_number  = cfg. cluster_min_pixel_number;
    Cluster.thrustable_thresold  = cfg.cluster_thrustable_thresold;
    Cluster.Voxel_revolusion  = cfg.cluster_Voxel_revolusion;
    Cluster.debug_en  = cfg. debug_en;
    Cluster.out_file  = cfg.cluster_out_file;
    hor_resolution_max  = cfg. hor_resolution_max;
    ver_resolution_max  = cfg. ver_resolution_max;
    buffer_dur  = cfg. buffer_dur;
    point_index  = cfg. point_index;
    frame_id  = cfg. frame_id;
    time_file = "";
    time_breakdown_file = "";
    max_ind   = floor(3.1415926 * 2 / hor_resolution_max);
    if (pcl_his_list.size() == 0)
    {   
        PointCloudXYZI::Ptr first_frame(new PointCloudXYZI());
        first_frame->reserve(400000);
        pcl_his_list.push_back(first_frame);
        laserCloudSteadObj_hist = PointCloudXYZI::Ptr(new PointCloudXYZI());
        laserCloudSteadObj = PointCloudXYZI::Ptr(new PointCloudXYZI());
        laserCloudDynObj = PointCloudXYZI::Ptr(new PointCloudXYZI());
        laserCloudDynObj_world = PointCloudXYZI::Ptr(new PointCloudXYZI());
        int xy_ind[3] = {-1, 1};
        for (int ind_hor = 0; ind_hor < 2*hor_num + 1; ind_hor ++)
        {
            for (int ind_ver = 0; ind_ver < 2*ver_num + 1; ind_ver ++)
            {
                pos_offset.push_back(((ind_hor)/2 + ind_hor%2)*xy_ind[ind_hor%2] * MAX_1D_HALF + ((ind_ver)/2 + ind_ver%2)*xy_ind[ind_ver%2]);
            }   
        }
    }
    map_cons_hor_num1 = ceil(map_cons_hor_thr1/hor_resolution_max);
    map_cons_ver_num1 = ceil(map_cons_ver_thr1/ver_resolution_max);
    interp_hor_num = ceil(interp_hor_thr/hor_resolution_max);
    interp_ver_num = ceil(interp_ver_thr/ver_resolution_max);
    map_cons_hor_num2 = ceil(map_cons_hor_thr2/hor_resolution_max);
    map_cons_ver_num2 = ceil(map_cons_ver_thr2/ver_resolution_max);
    occ_hor_num2 = ceil(occ_hor_thr2/hor_resolution_max);
    occ_ver_num2 = ceil(occ_ver_thr2/ver_resolution_max);
    depth_cons_hor_num2 = ceil(depth_cons_hor_thr2/hor_resolution_max);
    depth_cons_ver_num2 = ceil(depth_cons_ver_thr2/ver_resolution_max);
    map_cons_hor_num3 = ceil(map_cons_hor_thr3/hor_resolution_max);
    map_cons_ver_num3 = ceil(map_cons_ver_thr3/ver_resolution_max);
    occ_hor_num3 = ceil(occ_hor_thr3/hor_resolution_max);
    occ_ver_num3 = ceil(occ_ver_thr3/ver_resolution_max);
    depth_cons_hor_num3 = ceil(depth_cons_hor_thr3/hor_resolution_max);
    depth_cons_ver_num3 = ceil(depth_cons_ver_thr3/ver_resolution_max);
    buffer.init(buffer_size);

    pixel_fov_up = floor((fov_up/180.0*PI_MATH + 0.5 * PI_MATH)/ver_resolution_max);
    pixel_fov_down = floor((fov_down/180.0*PI_MATH + 0.5 * PI_MATH)/ver_resolution_max);
    pixel_fov_cut = floor((fov_cut/180.0*PI_MATH +  0.5 * PI_MATH)/ver_resolution_max);
    pixel_fov_left = floor((fov_left/180.0*PI_MATH +  PI_MATH)/hor_resolution_max);
    pixel_fov_right = floor((fov_right/180.0*PI_MATH +  PI_MATH)/hor_resolution_max);
    max_pointers_num = round((max_depth_map_num * depth_map_dur + buffer_delay)/frame_dur) + 1;
    point_soph_pointers.reserve(max_pointers_num);
    for (int i = 0; i < max_pointers_num; i++)
    {
        point_soph* p = new point_soph[points_num_perframe];
        point_soph_pointers.push_back(p);
    }
    if(time_file != "")
    {
        time_out.open(time_file, ios::out); 
    }
    if(time_breakdown_file != "")
    {
        time_breakdown_out.open(time_breakdown_file, ios::out); 
    }    
    Cluster.Init();
}

void  DynObjFilter::filter(PointCloudXYZI::Ptr feats_undistort, const M3D & rot_end, const V3D & pos_end, const double & scan_end_time)
{
    double t00 = omp_get_wtime();
    time_search = time_research = time_search_0 = time_build = time_total = time_other0 = 0.0;
    time_interp1 = time_interp2 =0;
    int num_build = 0, num_search_0 = 0, num_research = 0;
    if (feats_undistort == NULL) return;
    int size = feats_undistort->points.size();
    if (debug_en)
    {
        laserCloudSteadObj_hist.reset(new PointCloudXYZI());
        laserCloudSteadObj_hist->reserve(20*size);
    }
    dyn_tag_origin.clear();
    dyn_tag_origin.reserve(size);
    dyn_tag_origin.resize(size);
    dyn_tag_cluster.clear();
    dyn_tag_cluster.reserve(size);
    dyn_tag_cluster.resize(size);
    laserCloudDynObj.reset(new PointCloudXYZI());
    laserCloudDynObj->reserve(size);
    laserCloudDynObj_world.reset(new PointCloudXYZI());
    laserCloudDynObj_world->reserve(size);
    laserCloudSteadObj.reset(new PointCloudXYZI());
    laserCloudSteadObj->reserve(size);
    laserCloudDynObj_clus.reset(new PointCloudXYZI());
    laserCloudDynObj_clus->reserve(size);
    laserCloudSteadObj_clus.reset(new PointCloudXYZI());
    laserCloudSteadObj_clus->reserve(size); 
    ofstream out;
    ofstream out_origin;
    bool is_rec = false;
    bool is_rec_origin = false;
    if(is_set_path)
    {
        out.open(out_file, ios::out  | ios::binary);
        out_origin.open(out_file_origin, ios::out | ios::binary);
        if (out.is_open()) 
        {
            is_rec = true;
        }
        if (out_origin.is_open()) 
        {
            is_rec_origin = true;
        }
    }
    time_test1.reserve(size);
    time_test1.resize(size);
    time_test2.reserve(size);
    time_test2.resize(size);
    time_test3.reserve(size);
    time_test3.resize(size);
    time_occ_check.reserve(size);
    time_occ_check.resize(size);
    time_map_cons.reserve(size);
    time_map_cons.resize(size);
    time_proj.reserve(size);
    time_proj.resize(size);
    for(int i = 0; i < size; i++)
    {
        time_test1[i] = 0.0;
        time_test2[i] = 0.0;
        time_test3[i] = 0.0;
        time_occ_check[i] = 0.0;
        time_map_cons[i] = 0.0;
        time_proj[i] = 0.0;
    }
    int case2_num = 0;
    double t0 = omp_get_wtime();
    double time_case1 = 0, time_case2 = 0, time_case3 = 0;  
    pcl::PointCloud<MD_PointType> raw_points_world;
    raw_points_world.reserve(size);
    raw_points_world.resize(size);
    std::vector<int> index(size);
    for (int i = 0; i < size; i++) {
        index[i] = i;
    }
    vector<point_soph*> points;
    points.reserve(size);
    points.resize(size);
    point_soph* p = point_soph_pointers[cur_point_soph_pointers];
    if(time_file != "") time_out << size << " "; //rec computation time
    std::for_each(std::execution::par, index.begin(), index.end(), [&](const int &i)
    // std::for_each(std::execution::seq, index.begin(), index.end(), [&](const int &i)
    {   
         V3D p_body(feats_undistort->points[i].x, feats_undistort->points[i].y, feats_undistort->points[i].z);

        if (!std::isfinite(p_body(0)) || !std::isfinite(p_body(1)) || !std::isfinite(p_body(2)) || p_body.norm() > 500.0)
        {
            p[i].dyn = INVALID;
            dyn_tag_origin[i] = 0;
            dyn_tag_cluster[i] = -1;
            points[i] = &p[i];
            return; // пропускаем эту мусорную точку
        }

        p[i].reset();     
        int intensity = feats_undistort->points[i].curvature;
        V3D p_glob(rot_end * (p_body) + pos_end);
        p[i].glob = p_glob;        
        p[i].dyn = STATIC;
        p[i].rot = rot_end.transpose();
        p[i].transl = pos_end;
        p[i].time = scan_end_time;
        p[i].local = p_body;
        p[i].intensity = feats_undistort->points[i].intensity;
        if(dataset == 0 && fabs(intensity-666) < 10E-4)
        {
            p[i].is_distort = true;
        }
        if (InvalidPointCheck(p_body, intensity))
        {   
            p[i].dyn = INVALID;
            dyn_tag_origin[i] = 0;
            dyn_tag_cluster[i] = -1;
        }
        else if(SelfPointCheck(p_body, p[i].dyn))
        {
            p[i].dyn = INVALID;
            dyn_tag_origin[i] = 0;
        }
        else if (Case1(p[i]))
        {
            p[i].dyn = CASE1;
            dyn_tag_origin[i] = 1;
        }
        else if (Case2(p[i]))
        {
            p[i].dyn = CASE2;
            dyn_tag_origin[i] = 1;
        } 
        else if(Case3(p[i]))
        {
            p[i].dyn = CASE3;
            dyn_tag_origin[i] = 1;
        }
        else
        {
            dyn_tag_origin[i] = 0;
        }
        points[i] = &p[i];
    });   
    if(time_file != "") time_out << omp_get_wtime()-t0 << " "; //rec computation time 
    for(int i = 0; i < size; i++)
    {
        MD_PointType po;
        po.x = points[i]->local[0];
        po.y = points[i]->local[1];
        po.z = points[i]->local[2];
        po.intensity = points[i]->intensity;
        MD_PointType po_w;
        po_w.x = points[i]->glob[0];
        po_w.y = points[i]->glob[1];
        po_w.z = points[i]->glob[2];
        raw_points_world[i] = po;
        switch(points[i]->dyn)
        {
            case CASE1:
                po.normal_x = 1;
                laserCloudDynObj->push_back(po);
                laserCloudDynObj_world->push_back(po_w);
                break;
            case CASE2:
                po.normal_y = points[i]->occu_times;
                laserCloudDynObj->push_back(po);
                laserCloudDynObj_world->push_back(po_w);
                break;
            case CASE3:
                po.normal_z = points[i]->is_occu_times;
                laserCloudDynObj->push_back(po);
                laserCloudDynObj_world->push_back(po_w);
                break;
            default:
                laserCloudSteadObj->push_back(po_w);
        }
    }
	int num_1 = 0, num_2 = 0, num_3 = 0, num_inval = 0, num_neag = 0; 
    double clus_before = omp_get_wtime(); //rec computation time
    if (cluster_coupled || cluster_future)
    {
        Cluster.Clusterprocess(dyn_tag_cluster, *laserCloudDynObj, raw_points_world, scan_end_time, rot_end, pos_end);
        for(int i = 0; i < size; i++)
        {   
            MD_PointType po;
            po.x = points[i]->glob(0);
            po.y = points[i]->glob(1);
            po.z = points[i]->glob(2);
            po.curvature = i;
            switch (points[i]->dyn)
            {   
                case CASE1:               
                    if (dyn_tag_cluster[i] == 0) 
                    {
                        points[i]->dyn = STATIC;
                        points[i]->occu_times = -1;
                        points[i]->is_occu_times = -1;
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        laserCloudSteadObj_clus->push_back(po);
                        num_neag += 1;
                    }
                    else // case1
                    {
                        po.normal_x = 1;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        laserCloudDynObj_clus->push_back(po);
                        if(!dyn_filter_en)
                        {
                            po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        }
                        num_1 += 1;
                    }
                    break;               
                case CASE2:
                    if(dyn_tag_cluster[i] == 0)
                    {
                        points[i]->dyn = STATIC;
                        points[i]->occu_times = -1;
                        points[i]->is_occu_times = -1;
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        laserCloudSteadObj_clus->push_back(po);
                        num_neag += 1;
                    }
                    else
                    {
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        laserCloudDynObj_clus->push_back(po);
                        if(!dyn_filter_en)
                        {
                            po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        }
                        num_2 += 1;
                    }
                    break;
                case CASE3:
                    if(dyn_tag_cluster[i] == 0)
                    {
                        points[i]->dyn = STATIC;
                        points[i]->occu_times = -1;
                        points[i]->is_occu_times = -1;
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        laserCloudSteadObj_clus->push_back(po);
                        num_neag += 1;
                    }
                    else
                    {
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        laserCloudDynObj_clus->push_back(po);
                        if(!dyn_filter_en)
                        {
                            po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        }
                        num_3 += 1;
                    }
                    break;       
                case STATIC:
                    if(dyn_tag_cluster[i] == 1)
                    {
                        points[i]->dyn = CASE1;
                        points[i]->occu_times = -1;
                        points[i]->is_occu_times = -1;
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        laserCloudDynObj_clus->push_back(po);
                        if(!dyn_filter_en)
                        {
                            po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        }
                        num_1 += 1;
                    }
                    else
                    {
                        po.normal_x = 0;
                        po.normal_y = points[i]->is_occu_times;
                        po.normal_z = points[i]->occu_times;
                        po.intensity = (int) (points[i]->local.norm() * 10) + 10;
                        laserCloudSteadObj_clus->push_back(po);
                        num_neag += 1;
                    }
                    break;               
                default: //invalid
                    num_inval += 1;
                    break;
            }
        }
    }
    if(time_file != "") time_out << omp_get_wtime()-clus_before << " "; //rec computation time  
    double t3 = omp_get_wtime();
    Points2Buffer(points, index);
    double t4 = omp_get_wtime();
    if(time_file != "") time_out << omp_get_wtime()-t3 << " "; //rec computation time
    Buffer2DepthMap(scan_end_time);
    if(time_file != "") time_out << omp_get_wtime()-t3 << endl; //rec computation time
    if (cluster_coupled)
    {   
        for(int i = 0; i < size; i++)
        {
            if (dyn_tag_cluster[i] == 1)
            {
                if(is_rec) 
                {
                    int tmp = 251;
                    out.write((char*)&tmp, sizeof(int));
                }
            } 
            else
            {
                if(is_rec)
                {
                    int tmp = 9;
                    out.write((char*)&tmp, sizeof(int));
                } 
            } 

            if (dyn_tag_origin[i] == 1 || dyn_tag_origin[i] == 2)
            {
                if(is_rec_origin) 
                {
                    int tmp = 251;
                    out_origin.write((char*)&tmp, sizeof(int));
                }
            } 
            else
            {
                if(is_rec_origin)
                {
                    int tmp = 9;
                    out_origin.write((char*)&tmp, sizeof(int));
                }
            } 
        }
    }
    else
    {
        for(int i = 0; i < size; i++)
        {
            if (dyn_tag_origin[i] == 1 || dyn_tag_origin[i] == 2)
            {
                if(is_rec) 
                {
                    int tmp = 251;
                    out.write((char*)&tmp, sizeof(int));
                }
            } 
            else
            {
                if(is_rec)
                {
                    int tmp = 9;
                    out.write((char*)&tmp, sizeof(int));
                }
            } 
        }
    }
    double total_test1 = 0, total_test2 = 0, total_test3 = 0, total_proj = 0, total_occ = 0, total_map = 0;
    for (int i = 0; i < size; i++)
    {
        total_test1 += time_test1[i];
        total_test2 += time_test2[i];
        total_test3 += time_test3[i];
        total_proj += time_proj[i];
        total_occ += time_occ_check[i];
        total_map += time_map_cons[i];
    }
    if(time_breakdown_file != "") 
    {
        time_breakdown_out << total_test1 << " " << total_test2 << " "<< total_test3 << " " << total_proj << " " << total_occ << " " << total_map << endl;
    }
    frame_num_for_rec ++;
    cur_point_soph_pointers = (cur_point_soph_pointers + 1)%max_pointers_num;
    if(is_rec) out.close();
    time_total = omp_get_wtime() - t00;
    time_ind ++;
    time_total_avr = time_total_avr * (time_ind - 1) / time_ind + time_total / time_ind;
}

void  DynObjFilter::Points2Buffer(vector<point_soph*> &points, std::vector<int> &index_vector)
{
    int cur_tail = buffer.tail;
    buffer.push_parallel_prepare(points.size());
    std::for_each(std::execution::par, index_vector.begin(), index_vector.end(), [&](const int &i)
    {   
        buffer.push_parallel(points[i], cur_tail+i);
    });
}

void  DynObjFilter::Buffer2DepthMap(double cur_time)
{
    int len = buffer.size();
    double total_0 = 0.0;
    double total_1 = 0.0;
    double total_2 = 0.0;
    double total_3 = 0.0;
    double t = 0.0;
    int max_point = 0;
    for (int k = 0; k < len; k++)
    {   
        point_soph* point = buffer.front();
        if ((cur_time - point->time) >= buffer_delay - frame_dur/2.0)
        {   
            if(depth_map_list.size() == 0)
            {
                if(depth_map_list.size() < max_depth_map_num)
                {
                    map_index ++;
                    DepthMap::Ptr new_map_pointer(new DepthMap(point->rot, point->transl, point->time, map_index));
                    depth_map_list.push_back(new_map_pointer);
                }
                else
                {
                    buffer.pop();
                    continue;
                }          
            }
            else if((point->time - depth_map_list.back()->time) >= depth_map_dur - frame_dur/2.0)
            {
                map_index ++;
                if (depth_map_list.size() == max_depth_map_num)
                {   
                    depth_map_list.front()->Reset(point->rot, point->transl, point->time, map_index);
                    DepthMap::Ptr new_map_pointer = depth_map_list.front();
                    depth_map_list.pop_front();
                    depth_map_list.push_back(new_map_pointer);
                }
                else if (depth_map_list.size() < max_depth_map_num)
                {   
                    DepthMap::Ptr new_map_pointer(new DepthMap(point->rot, point->transl, point->time, map_index));
                    depth_map_list.push_back(new_map_pointer);
                }
            }
            switch (point->dyn)
            {
                if(depth_map_list.back()->depth_map.size() <= point->position) 
                case STATIC:
                    SphericalProjection(*point, depth_map_list.back()->map_index, depth_map_list.back()->project_R, depth_map_list.back()->project_T, *point);                  
                    if(depth_map_list.back()->depth_map[point->position].size() < max_pixel_points)
                    {
                        depth_map_list.back()->depth_map[point->position].push_back(point);
                        if (point->vec(2) > depth_map_list.back()->max_depth_all[point->position])  
                        {
                            depth_map_list.back()->max_depth_all[point->position] = point->vec(2);
                            depth_map_list.back()->max_depth_index_all[point->position] = depth_map_list.back()->depth_map[point->position].size()-1;
                        }
                        if (point->vec(2) < depth_map_list.back()->min_depth_all[point->position] ||\
                            depth_map_list.back()->min_depth_all[point->position] < 10E-5)  
                        {
                            depth_map_list.back()->min_depth_all[point->position] = point->vec(2);
                            depth_map_list.back()->min_depth_index_all[point->position] = depth_map_list.back()->depth_map[point->position].size()-1;
                        }
                        if (point->vec(2) < depth_map_list.back()->min_depth_static[point->position] ||\
                            depth_map_list.back()->min_depth_static[point->position] < 10E-5)  
                        {
                            depth_map_list.back()->min_depth_static[point->position] = point->vec(2);
                        }
                        if (point->vec(2) > depth_map_list.back()->max_depth_static[point->position])  
                        {
                            depth_map_list.back()->max_depth_static[point->position] = point->vec(2);
                        }                        
                    }
                    break;   
                case CASE1:   

                case CASE2:

                case CASE3:
                    SphericalProjection(*point, depth_map_list.back()->map_index, depth_map_list.back()->project_R, depth_map_list.back()->project_T, *point);
                    if(depth_map_list.back()->depth_map[point->position].size() < max_pixel_points)
                    {
                        depth_map_list.back()->depth_map[point->position].push_back(point);
                        if (point->vec(2) > depth_map_list.back()->max_depth_all[point->position])  
                        {
                            depth_map_list.back()->max_depth_all[point->position] = point->vec(2);
                            depth_map_list.back()->max_depth_index_all[point->position] = depth_map_list.back()->depth_map[point->position].size()-1;
                        }
                        if (point->vec(2) < depth_map_list.back()->min_depth_all[point->position] ||\
                            depth_map_list.back()->min_depth_all[point->position] < 10E-5)  
                        {
                            depth_map_list.back()->min_depth_all[point->position] = point->vec(2);
                            depth_map_list.back()->min_depth_index_all[point->position] = depth_map_list.back()->depth_map[point->position].size()-1;
                        }
                    }
                    break;
                default:    
                    break;
            }
            buffer.pop();
        }
        else
        {
            break;
        }
    }
    if (debug_en)
    {   
        for (int i = 0; i < depth_map_list.size(); i++)
        {
            for (int j = 0; j < depth_map_list[i]->depth_map.size(); j++)
            {
                for (int k = 0; k < depth_map_list[i]->depth_map[j].size(); k++)
                {
                    MD_PointType po;
                    point_soph* point = depth_map_list[i]->depth_map[j][k];
                    po.x = point->glob(0);
                    po.y = point->glob(1);
                    po.z = point->glob(2);
                    po.intensity = point->local(2);
                    po.curvature = point->local(1);
                    po.normal_x = point->hor_ind;
                    po.normal_y = point->ver_ind;
                    po.normal_z = point->dyn;
                    if(point->dyn == STATIC) laserCloudSteadObj_hist->push_back(po);
                }
            }
        }
    }
}

void  DynObjFilter::SphericalProjection(point_soph &p, int depth_index, const M3D &rot, const V3D &transl, point_soph &p_spherical)
{
    if(fabs(p.last_vecs.at(depth_index%HASH_PRIM)[2]) > 10E-5)
    {       
        p_spherical.vec = p.last_vecs.at(depth_index%HASH_PRIM);
        p_spherical.hor_ind = p.last_positions.at(depth_index%HASH_PRIM)[0];
        p_spherical.ver_ind = p.last_positions.at(depth_index%HASH_PRIM)[1];
        p_spherical.position = p.last_positions.at(depth_index%HASH_PRIM)[2];     
    }
    else
    {
        V3D p_proj(rot*(p.glob - transl));
        p_spherical.GetVec(p_proj, hor_resolution_max, ver_resolution_max);
        p.last_vecs.at(depth_index%HASH_PRIM) = p_spherical.vec;
        p.last_positions.at(depth_index%HASH_PRIM)[0] = p_spherical.hor_ind;
        p.last_positions.at(depth_index%HASH_PRIM)[1] = p_spherical.ver_ind;
        p.last_positions.at(depth_index%HASH_PRIM)[2] = p_spherical.position;
    }
}

bool  DynObjFilter::InvalidPointCheck(const V3D &body, const int intensity)
{
    if ((pow(body(0), 2) + pow(body(1), 2) + pow(body(2), 2)) < blind_dis*blind_dis || (dataset == 1 && fabs(body(0)) < 0.1 && fabs(body(1)) < 1.0) && fabs(body(2)) < 0.1)
    {
        return true;
    } 
    else
    {
        return false;
    }
}

bool  DynObjFilter::SelfPointCheck(const V3D &body, const dyn_obj_flg dyn)
{
    if (dataset == 0)
    {      
        if( (body(0) > -1.2 && body(0) < -0.4 && body(1) > -1.7 && body(1) < -1.0 && body(2) > -0.65 && body(2) < -0.4) || \
            (body(0) > -1.75 && body(0) < -0.85 && body(1) > 1.0 && body(1) < 1.6 && body(2) > -0.75 && body(2) < -0.40) || \
            (body(0) > 1.4 && body(0) < 1.7 && body(1) > -1.3 && body(1) < -0.9 && body(2) > -0.8 && body(2) < -0.6) || \
            (body(0) > 2.45 && body(0) < 2.6 && body(1) > -0.6 && body(1) < -0.45 && body(2) > -1.0 && body(2) < -0.9) || \
            (body(0) > 2.45 && body(0) < 2.6 && body(1) > 0.45 && body(1) < 0.6 && body(2) > -1.0 && body(2) < -0.9) )
        {
            return true;
        }
        else
        {
            return false;
        }
    } 
    return false;
}

bool  DynObjFilter::CheckVerFoV(const point_soph & p, const DepthMap &map_info)
{
    bool ver_up = false, ver_down = false;
    for(int i = p.ver_ind; i >= pixel_fov_down; i--)
    {   
        int cur_pos = p.hor_ind * MAX_1D_HALF + i;
        if(map_info.depth_map[cur_pos].size() > 0)
        {
            ver_down = true;
            break;
        }
    } 
    for(int i = p.ver_ind; i <= pixel_fov_up; i++)
    {   
        int cur_pos = p.hor_ind * MAX_1D_HALF + i;
        if(map_info.depth_map[cur_pos].size() > 0)
        {
            ver_up = true;
            break;
        }
    }   
    if(ver_up && ver_down)
    {
        return false;
    }
    else
    {
        return true;
    }
}

void DynObjFilter::CheckNeighbor(const point_soph & p, const DepthMap &map_info, float &max_depth, float &min_depth)
{   
    int n = checkneighbor_range;
    for (int i = -n; i <= n; i++)
    {
        for (int j = -n; j <= n; j++)
        {
            int cur_pos = (p.hor_ind + i) * MAX_1D_HALF + p.ver_ind + j;
            if(cur_pos < MAX_2D_N && cur_pos >= 0 && map_info.depth_map[cur_pos].size() > 0)
            {
                float cur_max_depth = map_info.max_depth_static[cur_pos];
                float cur_min_depth = map_info.min_depth_static[cur_pos];
                if(min_depth > 10E-5) min_depth = std::min(cur_min_depth, min_depth);
                else min_depth = cur_min_depth;
                if(max_depth > 10E-5) max_depth = std::max(cur_max_depth, max_depth);
                else max_depth = cur_max_depth;
            }
        }
    }
}

bool  DynObjFilter::Case1(point_soph & p)
{
    int depth_map_num = depth_map_list.size();
    int occluded_map = depth_map_num;
    for (int i = depth_map_num- 1; i >= 0; i--)
    {   
        SphericalProjection(p, depth_map_list[i]->map_index, depth_map_list[i]->project_R, depth_map_list[i]->project_T, p);
        if (fabs(p.hor_ind) > MAX_1D || fabs(p.ver_ind) > MAX_1D_HALF || p.vec(2) < 0.0f \
            || p.position < 0 || p.position >= MAX_2D_N)
        {
            p.dyn = INVALID;
            continue;
        }
        if (Case1Enter(p, *depth_map_list[i]))
        { 
            if (Case1FalseRejection(p, *depth_map_list[i]))
            {
                occluded_map -= 1;
            }
        }
        else
        {
            occluded_map -= 1;
        }
        if (occluded_map < occluded_map_thr1)
        {   
            return false;
        }
        if (occluded_map -(i) >= occluded_map_thr1)
        {
            return true;
        }
    }
    if(occluded_map >= occluded_map_thr1)
    {
        return true;
    }
    return false;  
}

bool  DynObjFilter::Case1Enter(const point_soph & p, const DepthMap &map_info)
{
    float max_depth = 0, min_depth = 0;
    float max_depth_all = 0, min_depth_all = 0;
    if (map_info.depth_map[p.position].size() > 0)
    {
        max_depth = map_info.max_depth_static[p.position];
        min_depth = map_info.min_depth_static[p.position];
    }
    else 
    {
        if(p.ver_ind <= pixel_fov_up && p.ver_ind >pixel_fov_down && \
           p.hor_ind <= pixel_fov_left && p.hor_ind >= pixel_fov_right && \
           CheckVerFoV(p, map_info))
        {
            CheckNeighbor(p, map_info, max_depth, min_depth);
        }
    }        
    float cur_min = max(cutoff_value, k_depth_min_thr1*(p.vec(2) - d_depth_min_thr1)) + enter_min_thr1;
    float cur_max = max(cutoff_value, k_depth_max_thr1*(p.vec(2) - d_depth_max_thr1)) + enter_max_thr1;
    float cur_depth = depth_thr1;
    if(dataset == 0 && p.is_distort) 
    {
        cur_min = enlarge_distort*cur_min;
        cur_max = enlarge_distort*cur_max;
        cur_depth = enlarge_distort*cur_depth;
    }
    if (p.vec(2) < min_depth - cur_max || \
        (min_depth < p.vec(2) - cur_min && max_depth > p.vec(2) + cur_max)||\
        (stop_object_detect && min_depth <10E-5 && max_depth < 10E-5 && map_info.depth_map[p.position].size() > 0 && p.vec(2) < map_info.max_depth_all[p.position] + 1.0)
        )
    {
        case1_num ++;
        return true;
    }
    return false;
}

bool  DynObjFilter::Case1FalseRejection(point_soph & p, const DepthMap &map_info)
{
    return Case1MapConsistencyCheck(p, map_info, case1_interp_en);
}

bool  DynObjFilter::Case1MapConsistencyCheck(point_soph & p, const DepthMap &map_info, bool interp)
{   
    float hor_half = max(map_cons_hor_dis1/(max(p.vec(2), blind_dis)), map_cons_hor_thr1);
    float ver_half = max(map_cons_ver_dis1/(max(p.vec(2), blind_dis)), map_cons_ver_thr1);
    float cur_map_cons_depth_thr1 = max(cutoff_value, k_depth_max_thr1*(p.vec(2) - d_depth_max_thr1)) + map_cons_depth_thr1;
    float cur_map_cons_min_thr1 = max(cutoff_value, k_depth_min_thr1*(p.vec(2) - d_depth_min_thr1)) + enter_min_thr1;
    float cur_map_cons_max_thr1 = max(cutoff_value, k_depth_max_thr1*(p.vec(2) - d_depth_max_thr1)) + enter_max_thr1;
    if(dataset == 0 && p.is_distort) 
    {
        cur_map_cons_depth_thr1 = enlarge_distort*cur_map_cons_depth_thr1;
        cur_map_cons_min_thr1 = enlarge_distort*cur_map_cons_min_thr1;
        cur_map_cons_max_thr1 = enlarge_distort*cur_map_cons_max_thr1;
    }
    if (fabs(p.vec(1)) < enlarge_z_thr1 / 57.3)
    {
        hor_half = enlarge_angle * hor_half;
        ver_half = enlarge_angle * ver_half;
        cur_map_cons_depth_thr1 = enlarge_depth * cur_map_cons_depth_thr1;
    }
    int cur_map_cons_hor_num1 = ceil(hor_half/hor_resolution_max);
    int cur_map_cons_ver_num1 = ceil(ver_half/ver_resolution_max);
    int num = 0;
    point_soph closest_point;
    float closest_dis = 100;
    for (int ind_hor = -cur_map_cons_hor_num1; ind_hor <= cur_map_cons_hor_num1; ind_hor ++)
    {
        for (int ind_ver = -cur_map_cons_ver_num1; ind_ver <= cur_map_cons_ver_num1; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];                        
            if (map_info.max_depth_static[pos_new] < p.vec(2) - cur_map_cons_min_thr1 || \
                map_info.min_depth_static[pos_new] > p.vec(2) + cur_map_cons_max_thr1)
            {
                continue;
            }   
            for (int j = 0; j < points_in_pixel.size(); j++)
            {
                const point_soph* point = points_in_pixel[j];
                if (point->dyn == STATIC &&\
                  (fabs(p.vec(2)-point->vec(2)) <  cur_map_cons_depth_thr1 ||\
                   ((p.vec(2)-point->vec(2)) >  cur_map_cons_depth_thr1 && (p.vec(2)-point->vec(2)) <  cur_map_cons_min_thr1)) && \
                    fabs(p.vec(0)-point->vec(0)) < hor_half && \
                  fabs(p.vec(1)-point->vec(1)) < ver_half)
                {
                    return true;
                }    
            }         
        }
    }   
    if(interp && (p.local(0) < self_x_b || p.local(0) > self_x_f || p.local(1) > self_y_l || p.local(1) < self_y_r))
    {
        float depth_static = DepthInterpolationStatic(p, map_info.map_index, map_info.depth_map);
        float cur_interp = interp_thr1;
        if(p.vec(2) > interp_start_depth1)
            cur_interp += ((p.vec(2) - interp_start_depth1)* interp_kp1 + interp_kd1);
        if(dataset == 0 )
        {
            if(fabs(depth_static+1) < 10E-5 || fabs(depth_static+2) < 10E-5)
            {
                return false;
            }
            else 
            {
                if(fabs(depth_static - p.vec(2)) < cur_interp)
                {
                    return true;
                } 
            }
        }
        else
        {
            if(fabs(depth_static+1) < 10E-5 || fabs(depth_static+2) < 10E-5)
            {
                return false;
            }
            else 
            {
                if(fabs(depth_static - p.vec(2)) < cur_interp)
                {
                    return true;
                } 
            }        
        }     
    }
    return false;
}

float DynObjFilter::DepthInterpolationStatic(point_soph & p, int map_index, const DepthMap2D &depth_map)
{
    if(fabs(p.last_depth_interps.at(map_index - depth_map_list.front()->map_index)) > 10E-4)
    {
        float depth_cal = p.last_depth_interps.at(map_index - depth_map_list.front()->map_index);
        return depth_cal;
    }
    V3F p_1 = V3F::Zero();
    V3F p_2 = V3F::Zero();
    V3F p_3 = V3F::Zero();
    vector<V3F> p_neighbors;
    int all_num = 0, static_num = 0, no_bg_num = 0;     
    for (int ind_hor = -interp_hor_num; ind_hor <= interp_hor_num; ind_hor ++)
    {
        for (int ind_ver = -interp_ver_num; ind_ver <= interp_ver_num; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF); 
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = depth_map[pos_new];     
            for (int j = 0; j < points_in_pixel.size(); j ++)
            {   
                const point_soph* point = points_in_pixel[j]; 
                if (fabs(point->time - p.time) < frame_dur)
                {
                    continue;
                }
                float hor_minus =  point->vec(0) - p.vec(0);
                float ver_minus =  point->vec(1) - p.vec(1);
                if (fabs(hor_minus) < interp_hor_thr && fabs(ver_minus) < interp_ver_thr)
                {
                    all_num ++;
                    if(point->dyn == STATIC) 
                    {
                        static_num ++;
                    }
                    if((point->vec(2) - p.vec(2)) <= interp_static_max && (p.vec(2) - point->vec(2)) < 5.0)
                    {
                        no_bg_num ++;
                    }
                    if(point->dyn == STATIC)
                    {
                        p_neighbors.push_back(point->vec);
                        if (p_1(2)<0.000001 || fabs(hor_minus) + fabs(ver_minus) < fabs(p_1(0) - p.vec(0)) + fabs(p_1(1) - p.vec(1)))
                        {
                            p_1 = point->vec;
                        }
                    }
                }
            }
        }
    }
    if (p_1(2)<10E-5)
    {
        p.last_depth_interps.at(map_index - depth_map_list.front()->map_index) = -1;
        return -1;
    }
    int cur_size = p_neighbors.size();
    for(int t_i = 0; t_i < cur_size-2; t_i++)
    {
        p_1 = p_neighbors[t_i];
        p_2 = V3F::Zero();
        p_3 = V3F::Zero();
        float min_fabs = 2*(interp_hor_thr + interp_ver_thr);
        float x = p.vec(0) - p_1(0);
        float y = p.vec(1) - p_1(1);
        float alpha  = 0, beta = 0;
        for(int i = t_i+1; i < cur_size-1; i++)
        {           
            if(fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1)) < min_fabs)
            {
                p_2 = p_neighbors[i];
                float single_fabs = fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1));
                if (single_fabs >= min_fabs) continue;
                for(int ii = i+1; ii < cur_size; ii++)
                {
                    float cur_fabs = fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1)) + \
                                    fabs(p_neighbors[ii](0)-p.vec(0)) + fabs(p_neighbors[ii](1)-p.vec(1));
                    if( cur_fabs < min_fabs)
                    {
                        float x1 = p_neighbors[i](0) - p_1(0);
                        float x2 = p_neighbors[ii](0) - p_1(0);
                        float y1 = p_neighbors[i](1) - p_1(1);
                        float y2 = p_neighbors[ii](1) - p_1(1);
                        float lower = x1*y2-x2*y1;
                        if(fabs(lower) > 10E-5)
                        {
                            alpha = (x*y2-y*x2)/lower;
                            beta = -(x*y1-y*x1)/lower;
                            if(alpha > 0 && alpha < 1 && beta > 0 && beta < 1 && (alpha + beta) > 0 && (alpha + beta) < 1)
                            {
                                p_3 = p_neighbors[ii];
                                min_fabs = cur_fabs; 
                            }
                            
                        }
                    }
                }  
            }
        }
        if (p_2(2)<10E-5 || p_3(2)<10E-5)
        {
            continue;
        }
        float depth_cal = (1-alpha-beta)*p_1(2) + alpha*p_2(2) + beta*p_3(2);
        p.last_depth_interps.at(map_index - depth_map_list.front()->map_index) = depth_cal;
        return depth_cal;
    }
    if(static_num > 0 && cur_size < all_num/2)
    {
        p.last_depth_interps.at(map_index - depth_map_list.front()->map_index) = -2;
        return -2;
    }
    else
    {
        p.last_depth_interps.at(map_index - depth_map_list.front()->map_index) = -2;
        return -2;
    }
}// return -1 denotes no point, -2 denotes no trianguolar but with points

bool  DynObjFilter::Case2(point_soph & p)
{   
    if(dataset == 0 && p.is_distort) return false;
    int first_i = depth_map_list.size();
    first_i -= 1;
    if(first_i < 0) return false;
    point_soph p_spherical = p;   
    SphericalProjection(p, depth_map_list[first_i]->map_index, depth_map_list[first_i]->project_R, depth_map_list[first_i]->project_T, p_spherical);  
    if (fabs(p_spherical.hor_ind) >= MAX_1D || fabs(p_spherical.ver_ind) >= MAX_1D_HALF || p_spherical.vec(2) < 0.0f || \
        p_spherical.position < 0 || p_spherical.position >= MAX_2D_N)
    {
        p.dyn = INVALID;
        return false;
    }
    int cur_occ_times = 0;
    if (Case2Enter(p_spherical, *depth_map_list[first_i]))
    {
        if (!Case2MapConsistencyCheck(p_spherical, *depth_map_list[first_i], case2_interp_en))
        {
            double ti = 0;
            float vi = 0; 
            float min_hor = occ_hor_thr2, min_ver = occ_ver_thr2;
            bool map_cons = true;    
            for (int ind_hor = -occ_hor_num2; ind_hor <= occ_hor_num2; ind_hor ++)
            {
                for (int ind_ver = -occ_ver_num2; ind_ver <= occ_ver_num2; ind_ver ++)
                {   
                    int pos_new = ((p_spherical.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p_spherical.ver_ind +ind_ver)%MAX_1D_HALF);       
                    if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
                    const vector<point_soph*> & points_in_pixel = depth_map_list[first_i]->depth_map[pos_new];                    
                    if (depth_map_list[first_i]->min_depth_all[pos_new] > p_spherical.vec(2))
                    {
                        continue;
                    }   
                    for (int k = 0; k < points_in_pixel.size() && map_cons; k++)
                    {
                        const point_soph*  p_occ = points_in_pixel[k];                   
                        if(Case2IsOccluded(p_spherical, *p_occ) && Case2DepthConsistencyCheck(*p_occ, *depth_map_list[first_i]))
                        {                        
                            cur_occ_times = 1;
                            if(cur_occ_times >= occluded_times_thr2) break;
                            ti = (p_occ->time + p.time)/2;
                            vi = (p_spherical.vec(2) - p_occ->vec(2))/(p.time - p_occ->time);
                            p.occu_index[0] = depth_map_list[first_i]->map_index;
                            p.occu_index[1] = pos_new;
                            p.occu_index[2] = k;
                            p.occ_vec = p_spherical.vec;
                            p.occu_times = cur_occ_times;
                            point_soph  p0 = p;
                            point_soph p1 = *points_in_pixel[k];                          
                            int i = depth_map_list.size();
                            i = i - 2;
                            V3D t1, t2;
                            t1.setZero();
                            t2.setZero();
                            while(i >= 0)
                            {                              
                                if(p1.occu_index[0] == -1 || p1.occu_index[0] < depth_map_list.front()->map_index)
                                {
                                    SphericalProjection(p1, depth_map_list[i]->map_index, depth_map_list[i]->project_R, depth_map_list[i]->project_T, p1);
                                    if(Case2SearchPointOccludingP(p1, *depth_map_list[i]))
                                    {
                                        p1.occ_vec = p1.vec;
                                    }
                                    else
                                    {
                                        break;
                                    }                                   
                                }                                
                                i = p1.occu_index[0]-depth_map_list.front()->map_index;
                                point_soph*  p2 = depth_map_list[i]->depth_map[p1.occu_index[1]][p1.occu_index[2]];                                   
                                SphericalProjection(p, depth_map_list[i]->map_index, depth_map_list[i]->project_R, depth_map_list[i]->project_T, p);
                                if(Case2MapConsistencyCheck(p, *depth_map_list[i], case2_interp_en))
                                {
                                    map_cons = false;
                                    break;
                                }
                                float vc = (p1.occ_vec(2) - p2->vec(2))/(p1.time - p2->time);
                                double tc = (p2->time + p1.time)/2;        
                                if (Case2IsOccluded(p, *p2) &&\
                                    Case2DepthConsistencyCheck(*p2, *depth_map_list[i]) && Case2VelCheck(vi, vc, ti-tc) )
                                {                            
                                    cur_occ_times += 1;
                                    if(cur_occ_times >= occluded_times_thr2)
                                    {
                                        p.occu_times = cur_occ_times;
                                        return true;
                                    }
                                    t2 = p2->glob;
                                    p1 = *p2;
                                    vi = vc;
                                    ti = tc;
                                }
                                else
                                {
                                    break;
                                }
                                i--;
                            }                       
                        } 
                        if(cur_occ_times >= occluded_times_thr2) break;
                    }
                    if(cur_occ_times >= occluded_times_thr2) break;
                }
                if(cur_occ_times >= occluded_times_thr2) break;
            }
        }
    }
    if (cur_occ_times >= occluded_times_thr2) 
    {
        p.occu_times = cur_occ_times;
        return true;
    }
    return false;
}

bool  DynObjFilter::Case2Enter(point_soph & p, const DepthMap &map_info)
{
    if(p.dyn != STATIC)
    {
        return false;
    }
    float max_depth = 0;
    float depth_thr2_final = max(cutoff_value, k_depth_max_thr2*(p.vec(2) - d_depth_max_thr2)) + occ_depth_thr2;
    if(map_info.depth_map[p.position].size() > 0)
    {
        int idx = map_info.max_depth_index_all[p.position];
        if (idx >= 0 && idx < map_info.depth_map[p.position].size())
        {
        const point_soph* max_point = map_info.depth_map[p.position][map_info.max_depth_index_all[p.position]];
        max_depth = max_point->vec(2); 
        float delta_t = (p.time - max_point->time);
        depth_thr2_final = min(depth_thr2_final, v_min_thr2*delta_t);
        }
    }
    if(p.vec(2) > max_depth + depth_thr2_final) 
    {
        case2_num ++;
        return true;
    }
    else
    {
        return false;
    }
}

bool  DynObjFilter::Case2MapConsistencyCheck(point_soph & p, const DepthMap &map_info, bool interp)
{
    float cur_hor = map_cons_hor_thr2;
    float cur_ver = map_cons_ver_thr2;
    float cur_depth = max(cutoff_value, k_depth_max_thr2*(p.vec(2) - d_depth_max_thr2)) + map_cons_depth_thr2;
    for (int ind_hor = -map_cons_hor_num2; ind_hor <= map_cons_hor_num2; ind_hor ++)
    {
        for (int ind_ver = -map_cons_ver_num2; ind_ver <= map_cons_ver_num2; ind_ver ++)  
        {
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];                         
            if (map_info.max_depth_all[pos_new] > p.vec(2) + cur_depth && \
                map_info.min_depth_all[pos_new] < p.vec(2) - cur_depth)
            {
                continue;
            }   
            for (int j = 0; j < points_in_pixel.size(); j++)
            {
                const point_soph* point = points_in_pixel[j];
                if (point->dyn == STATIC && \
                    fabs(p.time-point->time) > frame_dur && \
                    fabs(p.vec(2)-point->vec(2)) <  cur_depth && \
                    fabs(p.vec(0)-point->vec(0)) < map_cons_hor_thr2 && \
                    fabs(p.vec(1)-point->vec(1)) < map_cons_ver_thr2)
                {
                    return true;
                }               
            }         
        }
    }
    if(interp && (p.local(0) < self_x_b || p.local(0) > self_x_f || p.local(1) > self_y_l || p.local(1) < self_y_r) )
    {
        float cur_interp = interp_thr2*(depth_map_list.back()->map_index - map_info.map_index + 1);
        float depth_all = DepthInterpolationAll(p, map_info.map_index, map_info.depth_map);
        if( fabs(p.vec(2) - depth_all)  < cur_interp) 
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

bool  DynObjFilter::Case2SearchPointOccludingP(point_soph & p, const DepthMap &map_info)
{ 
    for (int ind_hor = -occ_hor_num2; ind_hor <= occ_hor_num2; ind_hor ++)
    {
        for (int ind_ver = -occ_ver_num2; ind_ver <= occ_ver_num2; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);         
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];            
            if (map_info.min_depth_all[pos_new] > p.vec(2))
            {
                continue;
            }   
            for (int j = 0; j < points_in_pixel.size(); j++)
            {
                const point_soph* p_cond = points_in_pixel[j];
                if (Case2IsOccluded(p, *p_cond) && Case2DepthConsistencyCheck(*p_cond, map_info)) 
                {
                    p.occu_index[0] = map_info.map_index;
                    p.occu_index[1] = pos_new;
                    p.occu_index[2] = j;
                    p.occ_vec = p.vec;
                    return true;
                }
            }        
        }
    }
    return false;
}

bool  DynObjFilter::Case2IsOccluded(const point_soph & p, const point_soph & p_occ)
{
    if((dataset == 0 && p_occ.is_distort) || (dataset == 0 && p.is_distort) || p_occ.dyn == INVALID) return false;
    if((p.local(0) > self_x_b && p.local(0) < self_x_f && p.local(1) < self_y_l && p.local(1) > self_y_r) || \
        (p_occ.local(0) > self_x_b && p_occ.local(0) < self_x_f && p_occ.local(1) < self_y_l && p_occ.local(1) > self_y_r))
    {
        return false;
    }
    float delta_t = p.time - p_occ.time;
    float cur_occ_hor = occ_hor_thr2; 
    float cur_occ_ver = occ_ver_thr2; 
    if(delta_t > 0)
    {
        float depth_thr2_final = min(max(cutoff_value, k_depth_max_thr2*(p.vec(2) - d_depth_max_thr2)) + occ_depth_thr2, v_min_thr2*delta_t);
        if (p.vec(2) >  p_occ.vec(2) + depth_thr2_final && \
            fabs(p.vec(0)-p_occ.vec(0)) < cur_occ_hor && \
            fabs(p.vec(1)-p_occ.vec(1)) < cur_occ_ver )
        {
            return true;
        }              
    }
    return false;
}

float DynObjFilter::DepthInterpolationAll(point_soph & p, int map_index, const DepthMap2D &depth_map)
{

    V3F p_1 = V3F::Zero();
    V3F p_2 = V3F::Zero();
    V3F p_3 = V3F::Zero();
    vector<V3F> p_neighbors;
    int all_num = 0;
    for (int ind_hor = -interp_hor_num; ind_hor <= interp_hor_num; ind_hor ++)
    {
        for (int ind_ver = -interp_ver_num; ind_ver <= interp_ver_num; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);          
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = depth_map[pos_new];           
            for (int j = 0; j < points_in_pixel.size(); j ++)
            {
                const point_soph*  point = points_in_pixel[j]; 
                if (fabs(point->time - p.time) < frame_dur)
                {
                    continue;
                }
                float hor_minus =  point->vec(0) - p.vec(0);
                float ver_minus =  point->vec(1) - p.vec(1);
                if (fabs(hor_minus) < interp_hor_thr && fabs(ver_minus) < interp_ver_thr)
                {
                    all_num ++;
                    p_neighbors.push_back(point->vec);
                    if (p_1(2)<0.000001 || fabs(hor_minus) + fabs(ver_minus) < fabs(p_1(0) - p.vec(0)) + fabs(p_1(1) - p.vec(1)))
                    {
                        p_1 = point->vec;
                    }                   
                }
            }
        }
    }
    int cur_size = p_neighbors.size();
    if (p_1(2)<10E-5 || cur_size < 3)
    {
        return -1;
    }    
    for(int t_i = 0; t_i < cur_size-2; t_i++)
    {
        p_1 = p_neighbors[t_i];
        p_2 = V3F::Zero();
        p_3 = V3F::Zero();
        float min_fabs = 2*(interp_hor_thr + interp_ver_thr);
        float x = p.vec(0) - p_1(0);
        float y = p.vec(1) - p_1(1);
        float alpha  = 0, beta = 0;
        for(int i = t_i+1; i < cur_size-1; i++)
        {
            if(fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1)) < min_fabs)
            {
                p_2 = p_neighbors[i];
                float single_fabs = fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1));
                if (single_fabs >= min_fabs) continue;
                for(int ii = i+1; ii < cur_size; ii++)
                {
                    float cur_fabs = fabs(p_neighbors[i](0)-p.vec(0)) + fabs(p_neighbors[i](1)-p.vec(1)) + \
                                    fabs(p_neighbors[ii](0)-p.vec(0)) + fabs(p_neighbors[ii](1)-p.vec(1));
                    if( cur_fabs < min_fabs)
                    {
                        float x1 = p_neighbors[i](0) - p_1(0);
                        float x2 = p_neighbors[ii](0) - p_1(0);
                        float y1 = p_neighbors[i](1) - p_1(1);
                        float y2 = p_neighbors[ii](1) - p_1(1);
                        float lower = x1*y2-x2*y1;
                        if(fabs(lower) > 10E-5)
                        {
                            alpha = (x*y2-y*x2)/lower;
                            beta = -(x*y1-y*x1)/lower;
                            if(alpha > 0 && alpha < 1 && beta > 0 && beta < 1 && (alpha + beta) > 0 && (alpha + beta) < 1)
                            {
                                p_3 = p_neighbors[ii];
                                min_fabs = cur_fabs; 
                            }                        
                        }
                    }
                }
            } 
        }
        if (p_2(2)<10E-5 || p_3(2)<10E-5)
        {
            continue;
        }
        float depth_cal = (1-alpha-beta)*p_1(2) + alpha*p_2(2) + beta*p_3(2);
        return depth_cal;
    }
    return -2;
} // -1 denotes no points, -2 denotes no triangular > 1000 denotes gauss interpolation 

bool  DynObjFilter::Case2DepthConsistencyCheck(const point_soph & p, const DepthMap &map_info)
{
    float all_minus = 0;
    int num = 0, smaller_num = 0, all_num = 0, greater_num = 0;
    for (int ind_hor = -depth_cons_hor_num2; ind_hor <= depth_cons_hor_num2; ind_hor ++)
    {
        for (int ind_ver = -depth_cons_ver_num2; ind_ver <= depth_cons_ver_num2; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF); 
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];
            for (int j = 0; j < points_in_pixel.size(); j ++)
            {
                const point_soph* point = points_in_pixel[j]; 
                if(fabs(point->time - p.time) < frame_dur && fabs(point->vec(0)-p.vec(0)) < depth_cons_hor_thr2 && \
                  fabs(point->vec(1)-p.vec(1)) < depth_cons_ver_thr2)
                {
                    all_num ++;
                    if (point->dyn == STATIC) 
                    {
                        float cur_minus = p.vec(2)-point->vec(2);
                        if (fabs(cur_minus) < depth_cons_depth_max_thr2)
                        {
                            num ++;
                            all_minus += fabs(point->vec(2)-p.vec(2));
                        }
                        else if (cur_minus > 0)
                        {
                            smaller_num ++;
                        }
                        else
                        {
                            greater_num ++;
                        }
                    }
                }
            }
        }
    }
    if(all_num > 0)
    {
        if(num > 1)
        {           
            float cur_depth_thr = max(depth_cons_depth_thr2, k_depth2*p.vec(2));
            if(all_minus/(num-1) > cur_depth_thr)
            {
                return false;
            }      
        }
        if(greater_num == 0 || smaller_num == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }  
}

bool  DynObjFilter::Case2VelCheck(float v1, float v2, double delta_t)
{   
    if(fabs(v1 - v2) < delta_t*acc_thr2)
    {
        return true;
    }
    return false;
}

bool  DynObjFilter::Case3VelCheck(float v1, float v2, double delta_t)
{
    if(fabs(v1 - v2) < delta_t*acc_thr3)
    {
        return true;
    }
    return false;
}

bool  DynObjFilter::Case3(point_soph & p)
{   
    if(dataset == 0 && p.is_distort) return false;
    int first_i = depth_map_list.size();
    first_i -= 1;
    if(first_i < 0) return false;
    point_soph p_spherical = p;
    SphericalProjection(p, depth_map_list[first_i]->map_index, depth_map_list[first_i]->project_R, depth_map_list[first_i]->project_T, p_spherical);  
    if (fabs(p_spherical.hor_ind) >= MAX_1D || fabs(p_spherical.ver_ind) >= MAX_1D_HALF || p_spherical.vec(2) < 0.0f || \
        p_spherical.position < 0 || p_spherical.position >= MAX_2D_N)
    {
        p.dyn = INVALID;
        return false;
    }
    int cur_occ_times = 0;
    if (Case3Enter(p_spherical, *depth_map_list[first_i]))
    {
        if (!Case3MapConsistencyCheck(p_spherical, *depth_map_list[first_i], case3_interp_en))
        {
            double ti = 0;
            float vi = 0; 
            float min_hor = occ_hor_thr3, min_ver = occ_ver_thr3;
            bool map_cons = true;       
            for (int ind_hor = -occ_hor_num3; ind_hor <= occ_hor_num3; ind_hor ++)
            {
                for (int ind_ver = -occ_ver_num3; ind_ver <= occ_ver_num3; ind_ver ++)
                {   
                    int pos_new = ((p_spherical.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p_spherical.ver_ind +ind_ver)%MAX_1D_HALF);  
                    if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
                    const vector<point_soph*> & points_in_pixel = depth_map_list[first_i]->depth_map[pos_new];                   
                    if (depth_map_list[first_i]->max_depth_all[pos_new] < p_spherical.vec(2))
                    {
                        continue;
                    }   
                    for (int k = 0; k < points_in_pixel.size() && map_cons; k++)
                    {
                        const point_soph* p_occ = points_in_pixel[k];                          
                        if(Case3IsOccluding(p_spherical, *p_occ) && Case3DepthConsistencyCheck(*p_occ, *depth_map_list[first_i]))
                        {
                            
                            cur_occ_times = 1;
                            ti = (p_occ->time + p.time)/2;
                            vi = (p_occ->vec(2) - p_spherical.vec(2))/(p.time - p_occ->time);
                            p.is_occu_index[0] = depth_map_list[first_i]->map_index;
                            p.is_occu_index[1] = pos_new;
                            p.is_occu_index[2] = k;
                            p.is_occ_vec = p_spherical.vec;
                            p.is_occu_times = cur_occ_times;                        
                            point_soph  p0 = p;
                            point_soph p1 = *points_in_pixel[k];                        
                            int i = depth_map_list.size();
                            i = i -2;
                            while(i >= 0)
                            {
                                if(p1.is_occu_index[0] == -1 || p1.is_occu_index[0] < depth_map_list.front()->map_index)
                                {
                                    SphericalProjection(p1, depth_map_list[i]->map_index, depth_map_list[i]->project_R, depth_map_list[i]->project_T, p1);
                                    if(Case3SearchPointOccludedbyP(p1, *depth_map_list[i]))
                                    {
                                        p1.is_occ_vec = p1.vec;
                                    }
                                    else
                                    {
                                        break;
                                    }               
                                }
                                i = p1.is_occu_index[0]-depth_map_list.front()->map_index;
                                point_soph* p2 = depth_map_list[i]->depth_map[p1.is_occu_index[1]][p1.is_occu_index[2]];                        
                                SphericalProjection(p, depth_map_list[i]->map_index, depth_map_list[i]->project_R, depth_map_list[i]->project_T, p);
                                if(Case3MapConsistencyCheck(p, *depth_map_list[i], case3_interp_en))
                                {
                                    map_cons = false;
                                    break;
                                }
                                float vc = -(p1.is_occ_vec(2) - p2->vec(2))/(p1.time - p2->time);
                                double tc = (p2->time + p1.time)/2;                               
                                if (Case3IsOccluding(p, *p2) &&\
                                    Case3DepthConsistencyCheck(*p2, *depth_map_list[i]) && Case3VelCheck(vi, vc, ti-tc) )
                                {                                    
                                    cur_occ_times += 1;
                                    if(cur_occ_times >= occluding_times_thr3)
                                    {
                                        p.is_occu_times = cur_occ_times;
                                        return true;
                                    }
                                    p1 = *p2;
                                    vi = vc;
                                    ti = tc;
                                }
                                else
                                {
                                    break;
                                }
                                i--;
                            }
                        } 
                        if(cur_occ_times >= occluding_times_thr3) break;
                    }
                    if(cur_occ_times >= occluding_times_thr3) break;
                }
                if(cur_occ_times >= occluding_times_thr3) break;
            }
        }
    }
    if (cur_occ_times >= occluding_times_thr3) 
    {
        p.is_occu_times = cur_occ_times;
        return true;
    }
    return false;
}

bool  DynObjFilter::Case3Enter(point_soph & p, const DepthMap &map_info)
{
    if(p.dyn != STATIC)
    {
        return false;
    }
    float min_depth = 0;
    float depth_thr3_final = max(cutoff_value, k_depth_max_thr3*(p.vec(2) - d_depth_max_thr3)) + occ_depth_thr3;
    if(map_info.depth_map[p.position].size() > 0)
    {
        int idx = map_info.min_depth_index_all[p.position];
        if (idx >= 0 && idx < map_info.depth_map[p.position].size())
        {
        const point_soph* min_point = map_info.depth_map[p.position][map_info.min_depth_index_all[p.position]];
        min_depth = min_point->vec(2); 
        float delta_t = (p.time - min_point->time);
        depth_thr3_final = min(depth_thr3_final, v_min_thr3*delta_t);
        }
    }
    if(dataset == 0 && p.is_distort)
    {
        depth_thr3_final = enlarge_distort*depth_thr3_final;
    }
    if(p.vec(2) < min_depth - depth_thr3_final)
    {
        case3_num ++;
        return true;
    }
    else
    {
        return false;
    }
}

bool  DynObjFilter::Case3MapConsistencyCheck(point_soph & p, const DepthMap &map_info, bool interp)
{
    float cur_v_min = v_min_thr3;
    float cur_hor = map_cons_hor_thr3;
    float cur_ver = map_cons_ver_thr3;
    float cur_depth = max(cutoff_value, k_depth_max_thr3*(p.vec(2) - d_depth_max_thr3)) + map_cons_depth_thr3;
    if(dataset == 0 && p.is_distort) cur_v_min = enlarge_distort*cur_v_min;
    for (int ind_hor = -map_cons_hor_num3; ind_hor <= map_cons_hor_num3; ind_hor ++)
    {
        for (int ind_ver = -map_cons_ver_num3; ind_ver <= map_cons_ver_num3; ind_ver ++)      
        {
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];                        
            if (map_info.max_depth_all[pos_new] > p.vec(2) + cur_depth && \
                map_info.min_depth_all[pos_new] < p.vec(2) - cur_depth)
            {
                continue;
            }   
            for (int j = 0; j < points_in_pixel.size(); j++)
            {
                const point_soph* point = points_in_pixel[j];
                if (point->dyn == STATIC &&
                    fabs(p.time-point->time) > frame_dur &&
                    (point->vec(2)-p.vec(2)) < cur_depth &&
                    fabs(p.vec(0)-point->vec(0)) < cur_hor &&
                    fabs(p.vec(1)-point->vec(1)) < cur_ver)
                {

                    return true;
                }               
            }         
        }
    }
    if(interp && (p.local(0) < self_x_b || p.local(0) > self_x_f || p.local(1) > self_y_l || p.local(1) < self_y_r))
    {
        float cur_interp = interp_thr3*(depth_map_list.back()->map_index - map_info.map_index + 1);
        float depth_all = DepthInterpolationAll(p, map_info.map_index, map_info.depth_map);
        if( fabs(p.vec(2) - depth_all)  < cur_interp) 
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    return false;
}

bool  DynObjFilter::Case3SearchPointOccludedbyP(point_soph & p, const DepthMap &map_info)
{
    for (int ind_hor = -occ_hor_num3; ind_hor <= occ_hor_num3; ind_hor ++)
    {
        for (int ind_ver = -occ_ver_num3; ind_ver <= occ_ver_num3; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);                
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];            
            if (map_info.min_depth_all[pos_new] > p.vec(2))
            {
                continue;
            }   
            for (int j = 0; j < points_in_pixel.size(); j++)
            {
                const point_soph* p_cond = points_in_pixel[j];
                if (Case3IsOccluding(p, *p_cond) && Case3DepthConsistencyCheck(*p_cond, map_info) ) 
                {
                    p.is_occu_index[0] = map_info.map_index;
                    p.is_occu_index[1] = pos_new;
                    p.is_occu_index[2] = j;
                    p.occ_vec = p.vec;
                    return true;
                }
            }        
        }
    }
    return false;
}

bool  DynObjFilter::Case3IsOccluding(const point_soph & p, const point_soph & p_occ)
{
    if((dataset == 0 && p_occ.is_distort) || (dataset == 0 && p.is_distort) || p_occ.dyn == INVALID) return false;
    if((p.local(0) > self_x_b && p.local(0) < self_x_f && p.local(1) < self_y_l && p.local(1) > self_y_r) || \
        (p_occ.local(0) > self_x_b && p_occ.local(0) < self_x_f && p_occ.local(1) < self_y_l && p_occ.local(1) > self_y_r))
    {
        return false;
    }
    float delta_t = p.time - p_occ.time;
    if(delta_t > 0)
    {
        float depth_thr3_final = min(max(cutoff_value, k_depth_max_thr3*(p.vec(2) - d_depth_max_thr3)) + map_cons_depth_thr3, v_min_thr3*delta_t);
        if(dataset == 0 && p.is_distort) depth_thr3_final = enlarge_distort*depth_thr3_final;
        if (p_occ.vec(2) > p.vec(2)  + depth_thr3_final && \
            fabs(p.vec(0)-p_occ.vec(0)) < occ_hor_thr3 && \
            fabs(p.vec(1)-p_occ.vec(1)) < occ_ver_thr3 )
        {
            return true;
        }            
    }
    return false;
}

bool  DynObjFilter::Case3DepthConsistencyCheck(const point_soph & p, const DepthMap &map_info)
{
    float all_minus = 0;
    int num = 0, smaller_num = 0, all_num = 0, greater_num = 0;//
    for (int ind_hor = -depth_cons_hor_num3; ind_hor <= depth_cons_hor_num3; ind_hor ++)
    {
        for (int ind_ver = -depth_cons_ver_num3; ind_ver <= depth_cons_ver_num3; ind_ver ++)
        {   
            int pos_new = ((p.hor_ind + ind_hor)%MAX_1D) * MAX_1D_HALF + ((p.ver_ind +ind_ver)%MAX_1D_HALF);      
            if (pos_new < 0 || pos_new >= MAX_2D_N)  continue;
            const vector<point_soph*> & points_in_pixel = map_info.depth_map[pos_new];
            for (int j = 0; j < points_in_pixel.size(); j ++)
            {
                const point_soph* point = points_in_pixel[j]; 
                if(fabs(point->time - p.time) < frame_dur && fabs(point->vec(0)-p.vec(0)) < depth_cons_hor_thr3 && \
                  fabs(point->vec(1)-p.vec(1)) < depth_cons_ver_thr3)
                {
                    all_num ++;
                    if (point->dyn == STATIC) 
                    {
                        float cur_minus = p.vec(2)-point->vec(2);
                        if (fabs(cur_minus) < depth_cons_depth_max_thr3)
                        {
                            num ++;
                            all_minus += fabs(point->vec(2)-p.vec(2));
                        }
                        else if (cur_minus > 0)
                        {
                            smaller_num ++;
                        }
                        else
                        {
                            greater_num ++;
                        }
                    }
                }
            }
        }
    }
    if(all_num > 0)
    {
        if(num > 1)
        {      
            float cur_depth_thr = max(depth_cons_depth_thr3, k_depth3*p.vec(2));
            if(all_minus/(num-1) > cur_depth_thr)
            {
                return false;
            }      
        }
        if(greater_num == 0 || smaller_num == 0)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        return false;
    }
}

void DynObjFilter::set_path(string file_path, string file_path_origin)
{
    is_set_path = true;
    out_file = file_path;
    out_file_origin = file_path_origin;
}

