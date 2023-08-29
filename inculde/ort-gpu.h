#pragma once
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>




struct predict_data{

    int intput_size;            //输入内存大小
    int output_size;            //输出内存大小
    std::vector<int64_t>  ip_Dim;     //输入维度
    std::vector<int64_t>  op_Dim;     //输出维度
    std::vector<int64_t> buffer_size;       //申请内存大小 
    //void* buffers_ptr[5];       //GPU指针
    float* intput;              //输入内存指针
    float* output;              //输出内存指针
};

extern const OrtApi* g_ort;
extern OrtValue* input_tensors;
extern OrtValue* output_tensors;
extern std::vector<std::string> input_names;
extern std::vector<std::string> output_names;
extern OrtSession* session;

bool ort_dml_init();
bool Free_DML(predict_data& pred_data);
