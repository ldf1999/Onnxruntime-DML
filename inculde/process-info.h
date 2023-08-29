#pragma once


bool Init_frame();


struct GridAndStride{
	int grid0;
	int grid1;
	int stride;
};

struct Porocess_info {
	std::vector<cv::Rect> boxes;	 //坐标容器
	std::vector<int> Classes;        //类别容器
	std::vector<float> confidences;  //置信度容器
	std::vector<int> indices;		 //解码后的信息
	std:: vector<float> recent;		 //点乘目标
	cv::Rect target_xywh;				 //最近目标的xywh	

	std::vector<int> strides = { 8, 16, 32 };       //步长
	std::vector<GridAndStride> grid_strides;    //网格
	int Num_Classes;

};	

struct alignas(float) BBOX{
	cv::Rect_<float> box;
	float confidence;  // bbox_conf * cls_conf        坐标回归的置信度 * 类别的置信度
	float index; //类别
};
