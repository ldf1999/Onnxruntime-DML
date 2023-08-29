#pragma once
#include<Windows.h>
#include "logging.h"
#include<opencv2/opencv.hpp>
#include<chrono>


//全局函数
bool Thread_Should_Stop();

struct global_info {

    cv::Mat img;                        //截取的图片
    int origin_x, origin_y,cx,cy;       //原点，中心点
    int window_width, window_height;    //窗口宽高
    int Input_Dim[4],Output_Dim[3];     //全局 输入/输出
    bool (*capture_map)();              //capture函数地址
    bool (*capture_free)();             //capture释放地址

    bool (*Pre_process)(cv::Mat& img);  //预处理函数地址
    bool (*Post_poress)();              //后处理函数地址

    bool (*Move_R)(int, int);           //移动函数地址
    bool (*Move_free)();                //释放地址
};

struct Time_info {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;

    std::chrono::system_clock::time_point capture_start;
    std::chrono::system_clock::time_point capture_end;

    std::chrono::system_clock::time_point Inference_start;
    std::chrono::system_clock::time_point Inference_end;
};

//全局变量
extern struct Config_Info cfg_info;         //cfg参数
extern struct global_info global_data;      //全局结构体
extern struct ort_dml_interface ort_dml;    //ort接口
extern struct predict_data pred_data;		//推理所需要的参数
extern struct Porocess_info Process_data;   //框架处理的数据

extern HANDLE Aim_Event;                         //事件
extern HANDLE Dynamic_Event;                         //事件


