#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>
#include <vector>
#include <numeric>

#include <ort-gpu.h>
#include "sf-info.h"
#include"config-info.h"
#include "process-info.h"

using namespace std;

//struct ort_dml_interface ort_dml;
struct predict_data pred_data;

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

OrtValue* input_tensors = nullptr;      //推理的输入
OrtValue* output_tensors = nullptr;     //推理的输出
std::vector<std::string> input_names;
std::vector<std::string> output_names;
OrtSession* session;
OrtRunOptions* run_options;   //添加一个新的全局变量

bool load_model(Ort::Env* env, OrtSession** session){
    char* ortModelStream;
    size_t size;

    std::ifstream file(cfg_info.Pred.engine_path.c_str(), std::ios::binary);
    if (!file.good()) {
        std::cout << "-------------------------------------------------------------" << std::endl;
        std::cerr << "读取 " << cfg_info.Pred.engine_path << " 错误, 请检查文件和路径." << std::endl;
        std::cout << "-------------------------------------------------------------" << std::endl;
        return false;
    }
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    ortModelStream = new char[size];
    assert(ortModelStream);
    file.read(ortModelStream, size);
    file.close();

    OrtSessionOptions* session_options;
    g_ort->CreateSessionOptions(&session_options);

    // 设置其他会话选项（如果需要）
    OrtStatus* status = g_ort->CreateSessionFromArray(*env, ortModelStream, size, session_options, session);
    if (status != nullptr) {
        std::cerr << "加载模型时发生错误: " << g_ort->GetErrorMessage(status) << std::endl;
        g_ort->ReleaseStatus(status);
        return false;
    }

    return true;
}

void prepare_input_data(OrtValue* input_tensor, const std::vector<int64_t>& input_dims, const cv::Mat& image) {
    float* input_tensor_data = nullptr;
    g_ort->GetTensorMutableData(input_tensor, reinterpret_cast<void**>(&input_tensor_data));
    size_t data_size = std::accumulate(input_dims.begin(), input_dims.end(), 1, std::multiplies<int64_t>());

    // 检查输入数据shape是否正确
    if (static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols) * static_cast<size_t>(image.channels()) != data_size) {
        std::cerr << "图像数据大小与输入tensor大小不匹配！" << std::endl;
        return;
    }

    // 预处理图像并将结果存储到 input_tensor_data 中
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_dims[2], input_dims[3])); // 根据需要调整输入图像大小
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F); // 将输入图像转换为浮点型

    // 图像格式转换，从HxWxC转换为ONNX运行时模型所期望的维度格式，如1xCxHxW
    for (size_t c = 0; c < static_cast<size_t>(img_float.channels()); c++) {
        for (size_t y = 0; y < static_cast<size_t>(img_float.rows); y++) {
            for (size_t x = 0; x < static_cast<size_t>(img_float.cols); x++) {
                size_t idx = c * img_float.rows * img_float.cols + y * img_float.cols + x;
                input_tensor_data[idx] = img_float.at<float>(cv::Vec3i(y, x, c));
            }
        }
    }
}

bool setup_input(OrtSession* session) {
    // 获取输入节点信息
    size_t num_input_nodes = 0;
    g_ort->SessionGetInputCount(session, &num_input_nodes);

    OrtAllocator* allocator = nullptr;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (allocator == nullptr) {
        std::cerr << "无法获取分配器" << std::endl;
        return false;
    }

    // 更新全局变量 input_names
    char* input_names_temp = nullptr;
    g_ort->SessionGetInputName(session, num_input_nodes - 1, allocator, &input_names_temp);
    input_names.push_back(input_names_temp);
    std::cout << "输入节点名称: " << input_names[0] << std::endl;
    g_ort->AllocatorFree(allocator, input_names_temp);

    // 获取输入维度信息
    size_t num_input_dims;
    OrtTypeInfo* input_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* input_tensor_info = nullptr;
    g_ort->SessionGetInputTypeInfo(session, num_input_nodes - 1, &input_typeinfo);
    g_ort->CastTypeInfoToTensorInfo(input_typeinfo, &input_tensor_info);
    g_ort->GetDimensionsCount(input_tensor_info, &num_input_dims);
    std::cout << "输入维度大小：" << num_input_dims << std::endl;
    std::vector<int64_t> input_dims;
    input_dims.resize(num_input_dims);
    g_ort->GetDimensions(input_tensor_info, input_dims.data(), num_input_dims);
    for (size_t i = 0; i < num_input_dims; i++)
        std::cout << "输入维度 " << i << " = " << input_dims[i] << std::endl;

    pred_data.ip_Dim = input_dims;

    // 创建输入tensor
    size_t input_tensor_size = input_dims[0] * input_dims[1] * input_dims[2] * input_dims[3];
    std::vector<float> input_tensor_values(input_tensor_size);

    // 获取输入tensor大小
    pred_data.intput_size = input_tensor_size;
    pred_data.buffer_size.push_back(input_tensor_size);


    // 创建内存信息
    OrtMemoryInfo* memory_info;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    // 创建输入张量
    g_ort->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_dims.data(), num_input_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensors);

    // 准备输入数据
    cv::Mat image = global_data.img; // 使用你的图像数据替换
    prepare_input_data(input_tensors, input_dims, image);

    // Release memory_info
    g_ort->ReleaseMemoryInfo(memory_info);

    g_ort->ReleaseTypeInfo(input_typeinfo);

    return true;
}

bool setup_output(OrtSession* session) {
    if (session == nullptr) {
        std::cerr << "Session is nullptr" << std::endl;
        return false;
    }

    // 获取输出节点信息
    size_t num_output_nodes = 0;
    g_ort->SessionGetOutputCount(session, &num_output_nodes);

    OrtAllocator* allocator = nullptr;
    g_ort->GetAllocatorWithDefaultOptions(&allocator);
    if (allocator == nullptr) {
        std::cerr << "无法获取分配器" << std::endl;
        return false;
    }

    // 更新全局变量 output_names
    char* output_names_temp = nullptr;
    g_ort->SessionGetOutputName(session, 0, allocator, &output_names_temp);
    output_names.push_back(output_names_temp);   // 在此行以下添加一行代码来复制 output_names_temp
    std::cout << "输出节点名称: " << output_names[0] << std::endl;
    g_ort->AllocatorFree(allocator, output_names_temp);

    // 获取输出维度信息
    size_t num_output_dims = 0;
    OrtTypeInfo* output_typeinfo = nullptr;
    const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;
    g_ort->SessionGetOutputTypeInfo(session, 0, &output_typeinfo);
    g_ort->CastTypeInfoToTensorInfo(output_typeinfo, &output_tensor_info);
    g_ort->GetDimensionsCount(output_tensor_info, &num_output_dims);

    std::cout << "输出维度大小：" << num_output_dims << std::endl;
    std::vector<int64_t> output_dims;
    output_dims.resize(num_output_dims);
    g_ort->GetDimensions(output_tensor_info, output_dims.data(), num_output_dims);
    for (size_t i = 0; i < num_output_dims; i++)
        std::cout << "输出维度 " << i << " = " << output_dims[i] << std::endl;
    pred_data.op_Dim = output_dims;

    // 获取内存信息对象
    OrtMemoryInfo* memory_info = nullptr;
    g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    auto element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    // 修复：使用 CreateTensorWithDataAsOrtValue 函数创建输出张量
    g_ort->CreateTensorWithDataAsOrtValue(memory_info, nullptr, 0, output_dims.data(), num_output_dims, element_type, &output_tensors);

    // 释放 output_typeinfo 和 memory_info
    g_ort->ReleaseTypeInfo(output_typeinfo);
    g_ort->ReleaseMemoryInfo(memory_info);

    return true;
}

bool ort_dml_init() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "My_Test");

    Ort::SessionOptions session_options;

    try {
        // 禁用内存模式和设置图优化级别
        session_options.DisableMemPattern();
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 设置执行模式和线程数
        session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        session_options.SetIntraOpNumThreads(1);

        // 添加 DML 执行提供程序
        OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);

        OrtSession* session = nullptr;
        if (!load_model(&env, &session)) {
            std::cerr << "加载模型失败" << std::endl;
            return false;
        }
        // 设置输入
        if (!setup_input(session)) {
            std::cerr << "设置输入失败" << std::endl;
            return false;
        }

        // 设置输出
        if (!setup_output(session)) {
            std::cerr << "设置输出失败." << std::endl;
            return false;
        }

        // 其他处理步骤...
    }
    catch (const Ort::Exception& ex) {
        std::cerr << "创建会话时发生错误: " << ex.what() << std::endl;
        return false;
    }
    // 如果执行成功，则输出提示信息
    std::cout << "Create Resource PASS." << std::endl;
    return true;
}

bool Free_DML()
{
    return false;
}

bool Free_DML(predict_data& pred_data) {
    // 释放结构体中的输出指针
    if (pred_data.output != nullptr) {
        delete[] pred_data.output;
        pred_data.output = nullptr;
    }
    // 释放输入张量
    if (input_tensors != nullptr) {
        g_ort->ReleaseValue(input_tensors);
        input_tensors = nullptr;
    }

    // 释放输出张量
    if (output_tensors != nullptr) {
        g_ort->ReleaseValue(output_tensors);
        output_tensors = nullptr;
    }

    // 清理保存的输入和输出节点名称
    input_names.clear();
    output_names.clear();

    // 如果有其他资源需要释放，请在此处添加相应的代码
    // ...

    return true;
}
