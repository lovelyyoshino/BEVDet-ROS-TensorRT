#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/functional.h>

#include "bevdet.h"
#include "bevpool.h"
#include "grid_sampler.cuh"

using std::chrono::duration;
using std::chrono::high_resolution_clock;

/**
 * @brief 构造函数用于初始化 BEVDet 对象，加载配置文件，初始化视角转换和引擎，并分配设备内存
 * @param  config_file      配置文件路径
 * @param  n_img           图像数量
 * @param  _cams_intrin     相机内参矩阵列表
 * @param  _cams2ego_rot   相机到自车坐标系的旋转矩阵列表
 * @param  _cams2ego_trans  相机到自车坐标系的平移矩阵列表
 * @param  imgstage_file    图像阶段的引擎文件路径
 * @param  bevstage_file    BEV阶段的引擎文件路径
 */
BEVDet::BEVDet(const std::string &config_file, int n_img,               
                        std::vector<Eigen::Matrix3f> _cams_intrin, 
                        std::vector<Eigen::Quaternion<float>> _cams2ego_rot, 
                        std::vector<Eigen::Translation3f> _cams2ego_trans,
                        const std::string &imgstage_file, 
                        const std::string &bevstage_file) : 
                        cams_intrin(_cams_intrin), 
                        cams2ego_rot(_cams2ego_rot), 
                        cams2ego_trans(_cams2ego_trans){
    // 初始化参数
    InitParams(config_file);
    
    if(n_img != N_img)//检查输入的图像数量是否匹配配置文件中的要求
    {
        printf("BEVDet need %d images, but given %d images!", N_img, n_img);
    }
    auto start = high_resolution_clock::now();
    
    // 初始化视角转换
    InitViewTransformer();
    auto end = high_resolution_clock::now();
    duration<float> t = end - start;
    printf("InitVewTransformer cost time : %.4lf ms\n", t.count() * 1000);

    InitEngine(imgstage_file, bevstage_file); // 初始化引擎，加载 imgstage_file 和 bevstage_file 文件
    MallocDeviceMemory();//分配设备内存
}

/**
///@brief 将当前相机的内参和外参转换并拷贝到GPU内存中
///@param  curr_cams2ego_rotMy 当前相机到自车坐标系的旋转矩阵列表
///@param  curr_cams2ego_transMy 当前相机到自车坐标系的平移矩阵列表
///@param  cur_cams_intrin  当前相机的内参矩阵列表
*/
void BEVDet::InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                       const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                       const std::vector<Eigen::Matrix3f> &cur_cams_intrin){
    // 分配主机内存来存储旋转矩阵、平移矩阵、内参矩阵、后旋转矩阵、后平移矩阵和变换矩阵
    float* rot_host = new float[N_img * 3 * 3];
    float* trans_host = new float[N_img * 3];
    float* intrin_host = new float[N_img * 3 * 3];
    float* post_rot_host = new float[N_img * 3 * 3];
    float* post_trans_host = new float[N_img * 3];
    float* bda_host = new float[3 * 3];

    // 旋转和内参都保存到数组里面
    for(int i = 0; i < N_img; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                rot_host[i * 9 + j * 3 + k] = curr_cams2ego_rot[i].matrix()(j, k);// 将旋转矩阵转换为数组
                intrin_host[i * 9 + j * 3 + k] = cur_cams_intrin[i](j, k);
                post_rot_host[i * 9 + j * 3 + k] = post_rot(j, k);
            }
            trans_host[i * 3 + j] = curr_cams2ego_trans[i].translation()(j);
            post_trans_host[i * 3 + j] = post_trans.translation()(j);
        }
    }

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i == j){
                bda_host[i * 3 + j] = 1.0;//这代表了相机坐标系到自车坐标系的变换矩阵
            }
            else{
                bda_host[i * 3 + j] = 0.0;
            }
        }
    }


    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["rot"]], rot_host, 
                                N_img * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));//将旋转矩阵拷贝到GPU内存中
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["trans"]], trans_host, 
                                N_img * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["intrin"]], intrin_host, 
                                N_img * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["post_rot"]], post_rot_host, 
                                N_img * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["post_trans"]], post_trans_host, 
                                N_img * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[imgbuffer_map["bda"]], bda_host, 
                                3 * 3 * sizeof(float), cudaMemcpyHostToDevice));

    delete[] rot_host;//删除指针
    delete[] trans_host;
    delete[] intrin_host;
    delete[] post_rot_host;
    delete[] post_trans_host;
    delete[] bda_host;
}
/**
///@brief 从配置文件中加载参数并初始化相关变量
///@param  config_file      配置文件路径
 */
void BEVDet::InitParams(const std::string &config_file)
{
    YAML::Node model_config = YAML::LoadFile(config_file);
    N_img = model_config["data_config"]["Ncams"].as<int>();//获取相机数量
    src_img_h = model_config["data_config"]["src_size"][0].as<int>();//获取原始图像的高度
    src_img_w = model_config["data_config"]["src_size"][1].as<int>();//获取原始图像的宽度
    input_img_h = model_config["data_config"]["input_size"][0].as<int>();//获取输入图像的高度
    input_img_w = model_config["data_config"]["input_size"][1].as<int>();//获取输入图像的宽度
    crop_h = model_config["data_config"]["crop"][0].as<int>();//获取裁剪的高度
    crop_w = model_config["data_config"]["crop"][1].as<int>();//获取裁剪的宽度
    mean.x = model_config["mean"][0].as<float>();//获取均值
    mean.y = model_config["mean"][1].as<float>();
    mean.z = model_config["mean"][2].as<float>();
    std.x = model_config["std"][0].as<float>();//获取标准差
    std.y = model_config["std"][1].as<float>();
    std.z = model_config["std"][2].as<float>();
    down_sample = model_config["model"]["down_sample"].as<int>();//下采样因子
    depth_start = model_config["grid_config"]["depth"][0].as<float>();//深度起始值
    depth_end = model_config["grid_config"]["depth"][1].as<float>();
    depth_step = model_config["grid_config"]["depth"][2].as<float>();
    x_start = model_config["grid_config"]["x"][0].as<float>();//网格配置信息
    x_end = model_config["grid_config"]["x"][1].as<float>();
    x_step = model_config["grid_config"]["x"][2].as<float>();
    y_start = model_config["grid_config"]["y"][0].as<float>();
    y_end = model_config["grid_config"]["y"][1].as<float>();
    y_step = model_config["grid_config"]["y"][2].as<float>();
    z_start = model_config["grid_config"]["z"][0].as<float>();
    z_end = model_config["grid_config"]["z"][1].as<float>();
    z_step = model_config["grid_config"]["z"][2].as<float>();
    bevpool_channel = model_config["model"]["bevpool_channels"].as<int>();//bevpool通道数
    nms_pre_maxnum = model_config["test_cfg"]["max_per_img"].as<int>();//上一阶段非极大值抑制的最大数量
    nms_post_maxnum = model_config["test_cfg"]["post_max_size"].as<int>();//非极大值抑制的最大数量
    score_thresh = model_config["test_cfg"]["score_threshold"].as<float>();//得分阈值
    nms_overlap_thresh = model_config["test_cfg"]["nms_thr"][0].as<float>();//非极大值抑制的阈值
    use_depth = model_config["use_depth"].as<bool>();//是否使用深度信息
    use_adj = model_config["use_adj"].as<bool>();//是否使用邻接帧
    
    if(model_config["sampling"].as<std::string>() == "bicubic"){//采样方式
        pre_sample = Sampler::bicubic;
    }
    else{
        pre_sample = Sampler::nearest;
    }

    std::vector<std::vector<float>> nms_factor_temp = model_config["test_cfg"]
                            ["nms_rescale_factor"].as<std::vector<std::vector<float>>>();//设置非极大值抑制的缩放因子
    nms_rescale_factor.clear();
    for(auto task_factors : nms_factor_temp){
        for(float factor : task_factors){
            nms_rescale_factor.push_back(factor);//每个任务的缩放因子，并将其展平存储到 out_num_task_head 中[1.0, 0.7, 0.7, 0.4, 0.55, 1.1, 1.0, 1.0, 1.5, 3.5]，这个数值反应了不同任务的缩放因子
        }
    }
    
    std::vector<std::vector<std::string>> class_name_pre_task;
    class_num = 0;
    YAML::Node tasks = model_config["model"]["tasks"];//读取任务类别信息
    class_num_pre_task = std::vector<int>();
    for(auto it : tasks){
        int num = it["num_class"].as<int>();
        class_num_pre_task.push_back(num);
        class_num += num;
        class_name_pre_task.push_back(it["class_names"].as<std::vector<std::string>>());//[car, truck, construction_vehicle, bus, trailer, barrier, motorcycle,bicycle, pedestrian, traffic_cone]
    }

    YAML::Node common_head_channel = model_config["model"]["common_head"]["channels"];//读取模型输出头的通道信息[2, 1, 3, 2, 2]
    YAML::Node common_head_name = model_config["model"]["common_head"]["names"];//读取模型输出头的名称信息[reg, height, dim, rot, vel]
    for(size_t i = 0; i< common_head_channel.size(); i++){
        out_num_task_head[common_head_name[i].as<std::string>()] = 
                                                        common_head_channel[i].as<int>();
    }

    resize_radio = (float)input_img_w / src_img_w;
    feat_h = input_img_h / down_sample;
    feat_w = input_img_w / down_sample;
    depth_num = (depth_end - depth_start) / depth_step;
    xgrid_num = (x_end - x_start) / x_step;
    ygrid_num = (y_end - y_start) / y_step;
    zgrid_num = (z_end - z_start) / z_step;
    bev_h = ygrid_num;
    bev_w = xgrid_num;

    // 初始化图像预处理过程中的旋转和平移矩阵
    post_rot << resize_radio, 0, 0,
                0, resize_radio, 0,
                0, 0, 1;
    post_trans.translation() << -crop_w, -crop_h, 0;

    // 初始化相邻帧的处理
    adj_num = 0;
    if(use_adj){
        adj_num = model_config["adj_num"].as<int>();
        adj_frame_ptr.reset(new adjFrame(adj_num, bev_h * bev_w, bevpool_channel));
    }


    postprocess_ptr.reset(new PostprocessGPU(class_num, score_thresh, nms_overlap_thresh,
                                            nms_pre_maxnum, nms_post_maxnum, down_sample,
                                            bev_h, bev_w, x_step, y_step, x_start,
                                            y_start, class_num_pre_task, nms_rescale_factor));//初始化后处理对象 PostprocessGPU

}

/**
///@brief 分配用于存储图像数据和阶段网络绑定数据的设备（GPU）内存。
 */
void BEVDet::MallocDeviceMemory(){
    CHECK_CUDA(cudaMalloc((void**)&src_imgs_dev, 
                                N_img * 3 * src_img_h * src_img_w * sizeof(uchar)));

    imgstage_buffer = (void**)new void*[imgstage_engine->getNbBindings()];//分配存储原始图像数据的设备内存 src_imgs_dev,getNbBindings这个函数返回引擎中绑定的数据的数量，关键点3
    for(int i = 0; i < imgstage_engine->getNbBindings(); i++){
        nvinfer1::Dims32 dim = imgstage_context->getBindingDimensions(i);//getBindingDimensions获取绑定的维度信息
        int size = 1;
        for(int j = 0; j < dim.nbDims; j++){//nbDims表示维度的数量
            size *= dim.d[j];//计算维度的乘积
        }
        size *= dataTypeToSize(imgstage_engine->getBindingDataType(i));//计算数据类型的大小，getBindingDataType获取绑定的数据类型
        CHECK_CUDA(cudaMalloc(&imgstage_buffer[i], size));//使用 cudaMalloc 分配内存
    }

    std::cout << "img num binding : " << imgstage_engine->getNbBindings() << std::endl;

    bevstage_buffer = (void**)new void*[bevstage_engine->getNbBindings()];//分配存储阶段网络绑定数据的设备内存 bevstage_buffer
    for(int i = 0; i < bevstage_engine->getNbBindings(); i++){
        nvinfer1::Dims32 dim = bevstage_context->getBindingDimensions(i);//获取绑定的维度信息
        int size = 1;
        for(int j = 0; j < dim.nbDims; j++){
            size *= dim.d[j];
        }
        size *= dataTypeToSize(bevstage_engine->getBindingDataType(i));//计算数据类型的大小
        CHECK_CUDA(cudaMalloc(&bevstage_buffer[i], size));//使用 cudaMalloc 分配内存
    }

    return;
}

/**
///@brief 初始化视角转换器，将激光雷达点云转换为 BEV 图像
 */
void BEVDet::InitViewTransformer(){

    int num_points = N_img * depth_num * feat_h * feat_w;
    Eigen::Vector3f* frustum = new Eigen::Vector3f[num_points];//初始化 frustum 点云数据结构

    for(int i = 0; i < N_img; i++){//遍历所有相机
        for(int d_ = 0; d_ < depth_num; d_++){//遍历深度
            for(int h_ = 0; h_ < feat_h; h_++){
                for(int w_ = 0; w_ < feat_w; w_++){
                    int offset = i * depth_num * feat_h * feat_w + d_ * feat_h * feat_w
                                                                 + h_ * feat_w + w_;//计算偏移量
                    (frustum + offset)->x() = (float)w_ * (input_img_w - 1) / (feat_w - 1);
                    (frustum + offset)->y() = (float)h_ * (input_img_h - 1) / (feat_h - 1);
                    (frustum + offset)->z() = (float)d_ * depth_step + depth_start;

                    // eliminate post transformation
                    *(frustum + offset) -= post_trans.translation();//将 frustum 中的点转换为原始图像坐标系
                    *(frustum + offset) = post_rot.inverse() * *(frustum + offset);//将 frustum 中的点转换为原始图像坐标系

                    //将转到的原始图像坐标系的点转换为相机坐标系
                    (frustum + offset)->x() *= (frustum + offset)->z();
                    (frustum + offset)->y() *= (frustum + offset)->z();
                    // img to ego -> rot -> trans
                    *(frustum + offset) = cams2ego_rot[i] * cams_intrin[i].inverse()
                                    * *(frustum + offset) + cams2ego_trans[i].translation();//将 frustum 中的点转换为世界坐标系

                    // voxelization,进行体素化处理
                    *(frustum + offset) -= Eigen::Vector3f(x_start, y_start, z_start);
                    (frustum + offset)->x() = (int)((frustum + offset)->x() / x_step);
                    (frustum + offset)->y() = (int)((frustum + offset)->y() / y_step);
                    (frustum + offset)->z() = (int)((frustum + offset)->z() / z_step);
                }
            }
        }
    }

    int* _ranks_depth = new int[num_points];//初始化深度和特征的排序
    int* _ranks_feat = new int[num_points];//初始化特征排序数组

    for(int i = 0; i < num_points; i++){
        _ranks_depth[i] = i;
    }
    for(int i = 0; i < N_img; i++){
        for(int d_ = 0; d_ < depth_num; d_++){
            for(int u = 0; u < feat_h * feat_w; u++){
                int offset = i * (depth_num * feat_h * feat_w) + d_ * (feat_h * feat_w) + u;
                _ranks_feat[offset] = i * feat_h * feat_w + u;//这里算出在多张图对应的像素点
            }
        }
    }

    //筛选有效点,即位于体素网格内部的点，并将其索引存储在 kept 向量中。
    std::vector<int> kept;
    for(int i = 0; i < num_points; i++){
        if((int)(frustum + i)->x() >= 0 && (int)(frustum + i)->x() < xgrid_num &&
           (int)(frustum + i)->y() >= 0 && (int)(frustum + i)->y() < ygrid_num &&
           (int)(frustum + i)->z() >= 0 && (int)(frustum + i)->z() < zgrid_num){
            kept.push_back(i);
        }
    }

    valid_feat_num = kept.size();
    int* ranks_depth_host = new int[valid_feat_num];
    int* ranks_feat_host = new int[valid_feat_num];
    int* ranks_bev_host = new int[valid_feat_num];
    int* order = new int[valid_feat_num];

    for(int i = 0; i < valid_feat_num; i++){
        Eigen::Vector3f &p = frustum[kept[i]];//将 frustum 中的世界坐标系点
        ranks_bev_host[i] = (int)p.z() * xgrid_num * ygrid_num + 
                            (int)p.y() * xgrid_num + (int)p.x();//计算在 BEV 图像中的位置
        order[i] = i;
    }

    thrust::sort_by_key(ranks_bev_host, ranks_bev_host + valid_feat_num, order);//按照 ranks_bev_host 排序。得到的是对应位置的索引
    for(int i = 0; i < valid_feat_num; i++){
        ranks_depth_host[i] = _ranks_depth[kept[order[i]]];
        ranks_feat_host[i] = _ranks_feat[kept[order[i]]];//获取排序后的点
    }

    delete[] _ranks_depth;
    delete[] _ranks_feat;
    delete[] frustum;
    delete[] order;

    //计算间隔的开始位置和长度
    std::vector<int> interval_starts_host;
    std::vector<int> interval_lengths_host;

    interval_starts_host.push_back(0);
    int len = 1;
    for(int i = 1; i < valid_feat_num; i++){
        if(ranks_bev_host[i] != ranks_bev_host[i - 1]){//如果两个点不在同一个体素网格内
            interval_starts_host.push_back(i);
            interval_lengths_host.push_back(len);
            len=1;
        }
        else{
            len++;
        }
    }
    
    interval_lengths_host.push_back(len);
    unique_bev_num = interval_lengths_host.size();
    //将数据拷贝到 GPU
    CHECK_CUDA(cudaMalloc((void**)&ranks_bev_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&ranks_depth_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&ranks_feat_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&interval_starts_dev, unique_bev_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&interval_lengths_dev, unique_bev_num * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(ranks_bev_dev, ranks_bev_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ranks_depth_dev, ranks_depth_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ranks_feat_dev, ranks_feat_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(interval_starts_dev, interval_starts_host.data(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(interval_lengths_dev, interval_lengths_host.data(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));

    // printf("Num_points : %d\n", num_points);
    // printf("valid_feat_num : %d\n", valid_feat_num);
    // printf("unique_bev_num : %d\n", unique_bev_num);
    // printf("valid rate : %.3lf\n", (float)valid_feat_num / num_points);

    delete[] ranks_bev_host;
    delete[] ranks_depth_host;
    delete[] ranks_feat_host;
}

/**
///@brief 打印维度信息
///@param  dim              维度信息
 */
void print_dim(nvinfer1::Dims dim){
    for(auto i = 0; i < dim.nbDims; i++){
        printf("%d%c", dim.d[i], i == dim.nbDims - 1 ? '\n' : ' ');
    }
}

/**
///@brief 初始化引擎，从文件反序列化 TensorRT 引擎，并创建执行上下文
///@param  imgstage_file    图像阶段的引擎文件路径
///@param  bevstage_file    BEV阶段的引擎文件路
///@return int 
 */
int BEVDet::InitEngine(const std::string &imgstage_file, const std::string &bevstage_file){
    if(DeserializeTRTEngine(imgstage_file, &imgstage_engine))//反序列化 TensorRT 引擎
    {
        return EXIT_FAILURE;
    }
    if(DeserializeTRTEngine(bevstage_file, &bevstage_engine)){
        return EXIT_FAILURE;
    }
    if(imgstage_engine == nullptr || bevstage_engine == nullptr){
        std::cerr << "Failed to deserialize engine file!" << std::endl;
        return EXIT_FAILURE;
    }
    imgstage_context = imgstage_engine->createExecutionContext();//创建执行上下文
    bevstage_context = bevstage_engine->createExecutionContext();

    if (imgstage_context == nullptr || bevstage_context == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context!" << std::endl;
        return EXIT_FAILURE;
    }

    // 设置绑定维度并初始化绑定映射
    imgstage_context->setBindingDimensions(0, 
                            nvinfer1::Dims32{4, {N_img, 3, input_img_h, input_img_w}});
    bevstage_context->setBindingDimensions(0,
            nvinfer1::Dims32{4, {1, bevpool_channel * (adj_num + 1), bev_h, bev_w}});
    imgbuffer_map.clear();
    for(auto i = 0; i < imgstage_engine->getNbBindings(); i++){
        auto dim = imgstage_context->getBindingDimensions(i);
        auto name = imgstage_engine->getBindingName(i);
        imgbuffer_map[name] = i;
        std::cout << name << " : ";
        print_dim(dim);

    }
    std::cout << std::endl;

    bevbuffer_map.clear();
    for(auto i = 0; i < bevstage_engine->getNbBindings(); i++){
        auto dim = bevstage_context->getBindingDimensions(i);
        auto name = bevstage_engine->getBindingName(i);
        bevbuffer_map[name] = i;
        std::cout << name << " : ";
        print_dim(dim);
    }    
    
    return EXIT_SUCCESS;
}

/**
///@brief 反序列化 TensorRT 引擎文件，并创建对应的 TensorRT 引擎对象
///@param  engine_file      引擎文件路径
///@param  engine_ptr      用于存储反序列化后的 TensorRT 引擎对象的指针
///@return int 
 */
int BEVDet::DeserializeTRTEngine(const std::string &engine_file, 
                                                    nvinfer1::ICudaEngine **engine_ptr){
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream engine_stream;
    engine_stream.seekg(0, engine_stream.beg);//从文件读取引擎数据

    std::ifstream file(engine_file);
    engine_stream << file.rdbuf();
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);//创建 TensorRT 运行时，关键点1
    if (runtime == nullptr) 
    {
        // std::string msg("Failed to build runtime parser!");
        // g_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        // std::cout << "" << "Failed to build runtime parser!" << std::endl;
        std::cout << "\033[1;31m" << "\nFailed to build runtime parser!\n" << "\033[0m" << std::endl;
        return EXIT_FAILURE;
    }
    engine_stream.seekg(0, std::ios::end);
    const int engine_size = engine_stream.tellg();

    engine_stream.seekg(0, std::ios::beg); //获取引擎数据的大小
    void* engine_str = malloc(engine_size);//分配内存
    engine_stream.read((char*)engine_str, engine_size);
    
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_str, engine_size, NULL);//使用 TensorRT 运行时反序列化引擎数据，关键点2
    if (engine == nullptr) 
    {
        // std::string msg("Failed to build engine parser!");
        // g_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());

        std::cout << "\033[1;31m" << "\nFailed to build engine parser!\n" << "\033[0m" << std::endl;

        return EXIT_FAILURE;
    }
    *engine_ptr = engine;//转为ptr指针传出去
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true){//检查反序列化是否成功，并打印引擎绑定信息
            printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
        }
        else{
            printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
        }
    }
    return EXIT_SUCCESS;
}

/**
///@brief 获取相邻帧的特征，并对其进行对齐
///@param  curr_scene_token 当前场景的标识符
///@param  ego2global_rot   自车坐标系到全局坐标系的旋转矩阵
///@param  ego2global_trans 自车坐标系到全局坐标系的平移向量
///@param  bev_buffer       存储 BEV 特征的缓冲区
 */
void BEVDet::GetAdjFrameFeature(const std::string &curr_scene_token, 
                         const Eigen::Quaternion<float> &ego2global_rot,
                         const Eigen::Translation3f &ego2global_trans,
                         float* bev_buffer) {
    /* bev_buffer : 720 * 128 x 128
    */
    bool reset = false;
    if(adj_frame_ptr->buffer_num == 0 || adj_frame_ptr->lastScenesToken() != curr_scene_token){//检查是否需要重置相邻帧缓冲区
        adj_frame_ptr->reset();
        for(int i = 0; i < adj_num; i++){
            adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, ego2global_rot,
                                                                        ego2global_trans);//保存当前帧的 BEV 特征
        }
        reset = true;
    }

    for(int i = 0; i < adj_num; i++){
        const float* adj_buffer = adj_frame_ptr->getFrameBuffer(i);

        Eigen::Quaternion<float> adj_ego2global_rot;
        Eigen::Translation3f adj_ego2global_trans;
        adj_frame_ptr->getEgo2Global(i, adj_ego2global_rot, adj_ego2global_trans);//根据索引获取相邻帧的自车坐标系到全局坐标系的变换矩阵

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        AlignBEVFeature(ego2global_rot, adj_ego2global_rot, ego2global_trans,
                        adj_ego2global_trans, adj_buffer, 
                        bev_buffer + (i + 1) * bev_w * bev_h * bevpool_channel, stream);//获取相邻帧的 BEV 特征，并进行对齐
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaStreamDestroy(stream));
    }


    if(!reset){
        adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, 
                                                    ego2global_rot, ego2global_trans);
    }
}

/**
///@brief 对齐 BEV 特征，使相邻帧的特征在同一全局坐标系下对齐
///@param  curr_ego2global_rotMy 当前帧的自车坐标系到全局坐标系的旋转矩阵
///@param  adj_ego2global_rotMy 相邻帧的自车坐标系到全局坐标系的旋转矩阵
///@param  curr_ego2global_transMy 当前帧的自车坐标系到全局坐标系的平移向量
///@param  adj_ego2global_transMy 相邻帧的自车坐标系到全局坐标系的平移向量
///@param  input_bev         输入的 BEV 特征
///@param  output_bev       输出的对齐后的 BEV 特征
///@param  stream          CUDA 流，用于异步操
 */
void BEVDet::AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,
                             const Eigen::Quaternion<float> &adj_ego2global_rot,
                             const Eigen::Translation3f &curr_ego2global_trans,
                             const Eigen::Translation3f &adj_ego2global_trans,
                             const float* input_bev,
                             float* output_bev,
                             cudaStream_t stream){
    Eigen::Matrix4f curr_e2g_transform;
    Eigen::Matrix4f adj_e2g_transform;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            curr_e2g_transform(i, j) = curr_ego2global_rot.matrix()(i, j);
            adj_e2g_transform(i, j) = adj_ego2global_rot.matrix()(i, j);
        }
    }
    for(int i = 0; i < 3; i++){
        curr_e2g_transform(i, 3) = curr_ego2global_trans.vector()(i);
        adj_e2g_transform(i, 3) = adj_ego2global_trans.vector()(i);

        curr_e2g_transform(3, i) = 0.0;
        adj_e2g_transform(3, i) = 0.0;
    }
    curr_e2g_transform(3, 3) = 1.0;
    adj_e2g_transform(3, 3) = 1.0;

    Eigen::Matrix4f currEgo2adjEgo = adj_e2g_transform.inverse() * curr_e2g_transform;//计算当前帧和相邻帧的自车到全局变换矩阵
    Eigen::Matrix3f currEgo2adjEgo_2d;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            currEgo2adjEgo_2d(i, j) = currEgo2adjEgo(i, j);
        }
    }
    currEgo2adjEgo_2d(2, 0) = 0.0;
    currEgo2adjEgo_2d(2, 1) = 0.0;
    currEgo2adjEgo_2d(2, 2) = 1.0;
    currEgo2adjEgo_2d(0, 2) = currEgo2adjEgo(0, 3);
    currEgo2adjEgo_2d(1, 2) = currEgo2adjEgo(1, 3);

    Eigen::Matrix3f gridbev2egobev;
    gridbev2egobev(0, 0) = x_step;
    gridbev2egobev(1, 1) = y_step;
    gridbev2egobev(0, 2) = x_start;
    gridbev2egobev(1, 2) = y_start;
    gridbev2egobev(2, 2) = 1.0;

    gridbev2egobev(0, 1) = 0.0;
    gridbev2egobev(1, 0) = 0.0;
    gridbev2egobev(2, 0) = 0.0;
    gridbev2egobev(2, 1) = 0.0;

    Eigen::Matrix3f currgrid2adjgrid = gridbev2egobev.inverse() * currEgo2adjEgo_2d * gridbev2egobev;//计算当前帧到相邻帧的变换矩阵


    float* grid_dev;
    float* transform_dev;
    CHECK_CUDA(cudaMalloc((void**)&grid_dev, bev_h * bev_w * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&transform_dev, 9 * sizeof(float)));


    CHECK_CUDA(cudaMemcpy(transform_dev, Eigen::Matrix3f(currgrid2adjgrid.transpose()).data(), 
                                                        9 * sizeof(float), cudaMemcpyHostToDevice));//将变换矩阵拷贝到设备内存

    compute_sample_grid_cuda(grid_dev, transform_dev, bev_w, bev_h, stream);//计算采样网格


    int output_dim[4] = {1, bevpool_channel, bev_w, bev_h};
    int input_dim[4] = {1, bevpool_channel, bev_w, bev_h};
    int grid_dim[4] = {1, bev_w, bev_h, 2};
    

    grid_sample(output_bev, input_bev, grid_dev, output_dim, input_dim, grid_dim, 4,
                GridSamplerInterpolation::Bilinear, GridSamplerPadding::Zeros, true, stream);//将变换矩阵应用于 BEV 特征，生成对齐后的 BEV 特征
    CHECK_CUDA(cudaFree(grid_dev));
    CHECK_CUDA(cudaFree(transform_dev));
}



/**
///@brief 进行 BEV 检测推理，包括图像预处理、前向传播、特征对齐和后处理，并输出检测结果和时间消耗
///@param  cam_data         包含相机数据和参数的结构体
///@param  out_detections   用于存储检测结果的容器
///@param  cost_time        用于存储推理总时间的变量
///@param  idx              推理的索引，用于打印调试信息
///@return int 
 */
int BEVDet::DoInfer(const camsData& cam_data, std::vector<Box> &out_detections, float &cost_time,
                                                                                    int idx)
                                                                                    {
    
    printf("-------------------%d-------------------\n", idx + 1);

    printf("scenes_token : %s, timestamp : %lld\n", cam_data.param.scene_token.data(), 
                                cam_data.param.timestamp);

    auto pre_start = high_resolution_clock::now();
    // [STEP 1] : preprocess image, including resize, crop and normalize

    // 6张图像数据拷贝到gpu上src_imgs_dev
    CHECK_CUDA(cudaMemcpy(src_imgs_dev, cam_data.imgs_dev, 
        N_img * src_img_h * src_img_w * 3 * sizeof(uchar), cudaMemcpyDeviceToDevice));
    
    // 预处理
    preprocess(src_imgs_dev, (float*)imgstage_buffer[imgbuffer_map["images"]], N_img, src_img_h, src_img_w,
        input_img_h, input_img_w, resize_radio, resize_radio, crop_h, crop_w, mean, std, pre_sample);//包括图像数据的拷贝、图像预处理（如调整尺寸、裁剪和归一化）以及初始化深度信息

    // 初始化深度，将处理好的值放在imgstage_buffer中
    InitDepth(cam_data.param.cams2ego_rot, cam_data.param.cams2ego_trans, cam_data.param.cams_intrin);//使用 TensorRT 进行图像阶段的推理

    CHECK_CUDA(cudaDeviceSynchronize());//等待所有的流操作完成

    // 预处理时间
    auto pre_end = high_resolution_clock::now();

    // [STEP 2] : image stage network forward
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));//创建流
    
    if(!imgstage_context->enqueueV2(imgstage_buffer, stream, nullptr))//使用 TensorRT 进行图像阶段的推理，对应图像engine，处理完会返回到imgstage_buffer中，stream作用是异步执行
    {
        printf("Image stage forward failing!\n");
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto imgstage_end = high_resolution_clock::now();


    // [STEP 3] : bev pool
    // size_t id1 = use_depth ? 7 : 1;
    // size_t id2 = use_depth ? 8 : 2;

    bev_pool_v2(bevpool_channel, unique_bev_num, bev_h * bev_w,
                (float*)imgstage_buffer[imgbuffer_map["depth"]], 
                (float*)imgstage_buffer[imgbuffer_map["images_feat"]], 
                ranks_depth_dev, ranks_feat_dev, ranks_bev_dev,
                interval_starts_dev, interval_lengths_dev,
                (float*)bevstage_buffer[bevbuffer_map["BEV_feat"]]

                );//对预处理后的特征进行 BEV 池化操作
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto bevpool_end = high_resolution_clock::now();


    // [STEP 4] : align BEV feature

    if(use_adj){
        GetAdjFrameFeature(cam_data.param.scene_token, cam_data.param.ego2global_rot, 
                        cam_data.param.ego2global_trans, (float*)bevstage_buffer[bevbuffer_map["BEV_feat"]]);//如果使用相邻帧数据，则进行特征对齐
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto align_feat_end = high_resolution_clock::now();


    // [STEP 5] : BEV stage network forward
    if(!bevstage_context->enqueueV2(bevstage_buffer, stream, nullptr)){
        printf("BEV stage forward failing!\n");
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto bevstage_end = high_resolution_clock::now();


    // [STEP 6] : postprocess 使用 TensorRT 进行 BEV 阶段的推理

    postprocess_ptr->DoPostprocess(bevstage_buffer, out_detections);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto post_end = high_resolution_clock::now();

    // release stream
    CHECK_CUDA(cudaStreamDestroy(stream));

    duration<double> pre_t = pre_end - pre_start;
    duration<double> imgstage_t = imgstage_end - pre_end;
    duration<double> bevpool_t = bevpool_end - imgstage_end;
    duration<double> align_t = duration<double>(0);
    duration<double> bevstage_t;
    if(use_adj)
    {
        align_t = align_feat_end - bevpool_end;
        bevstage_t = bevstage_end - align_feat_end;
    }
    else{
        bevstage_t = bevstage_end - bevpool_end;
    }
    duration<double> post_t = post_end - bevstage_end;

    printf("[Preprocess   ] cost time: %5.3lf ms\n", pre_t.count() * 1000);
    printf("[Image stage  ] cost time: %5.3lf ms\n", imgstage_t.count() * 1000);
    printf("[BEV pool     ] cost time: %5.3lf ms\n", bevpool_t.count() * 1000);
    if(use_adj){
        printf("[Align Feature] cost time: %5.3lf ms\n", align_t.count() * 1000);
    }
    printf("[BEV stage    ] cost time: %5.3lf ms\n", bevstage_t.count() * 1000);
    printf("[Postprocess  ] cost time: %5.3lf ms\n", post_t.count() * 1000);

    duration<double> sum_time = post_end - pre_start;
    cost_time = sum_time.count() * 1000;
    printf("[Infer total  ] cost time: %5.3lf ms\n", cost_time);

    printf("Detect %ld objects\n", out_detections.size());
    return EXIT_SUCCESS;
}

BEVDet::~BEVDet(){
    CHECK_CUDA(cudaFree(ranks_bev_dev));
    CHECK_CUDA(cudaFree(ranks_depth_dev));
    CHECK_CUDA(cudaFree(ranks_feat_dev));
    CHECK_CUDA(cudaFree(interval_starts_dev));
    CHECK_CUDA(cudaFree(interval_lengths_dev));

    CHECK_CUDA(cudaFree(src_imgs_dev));

    for(int i = 0; i < imgstage_engine->getNbBindings(); i++){
        CHECK_CUDA(cudaFree(imgstage_buffer[i]));
    }
    delete[] imgstage_buffer;

    for(int i = 0; i < bevstage_engine->getNbBindings(); i++){
        CHECK_CUDA(cudaFree(bevstage_buffer[i]));
    }
    delete[] bevstage_buffer;

    imgstage_context->destroy();
    bevstage_context->destroy();

    imgstage_engine->destroy();
    bevstage_engine->destroy();

}


__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType)
{
    switch ((int)dataType)
    {
    case int(nvinfer1::DataType::kFLOAT):
        return 4;
    case int(nvinfer1::DataType::kHALF):
        return 2;
    case int(nvinfer1::DataType::kINT8):
        return 1;
    case int(nvinfer1::DataType::kINT32):
        return 4;
    case int(nvinfer1::DataType::kBOOL):
        return 1;
    default:
        return 4;
    }
}
