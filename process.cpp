

#include "sf-info.h"
#include "config-info.h"
#include "process-info.h"
#include "ort-gpu.h"

// --------------------------------- 预处理--------------------------------- //
// ---- yolov5 ----//
bool Yolov5_Pre_Processing(cv::Mat& img) {

	/*for (int c = 0; c < 3; ++c) { //1-2 最高3ms ，最低
		for (int h = 0; h < img.rows; ++h){
			//获取第i行首像素指针 
			cv::Vec3b* p1 = img.ptr<cv::Vec3b>(h);
			for (int w = 0; w < img.cols; ++w){
				pred_data.intput[c * img.cols * img.rows + h * img.cols + w] = (p1[w][c]) / 255.0f;   //yolov5
			}
		}
	}*/
	int i = 0;
	for (int row = 0; row < 416; ++row)
	{
		uchar* uc_pixel = img.data + row * img.step;  //获取img全部通道的第row行数据
		for (int col = 0; col < 416; ++col)       //将row行的数据赋值给data的对应位置
		{

			pred_data.intput[i] = (float)uc_pixel[2] / 255.0;   //第1通道
			pred_data.intput[i + 416 * 416] = (float)uc_pixel[1] / 255.0;   //第2通道
			pred_data.intput[i + 2 * 416 * 416] = (float)uc_pixel[0] / 255.0;   //第三通道
			uc_pixel += 3;
			++i;
		}
	}
	//std::cout << "YoloV5 Pre PASS..." << std::endl;
	return true;
}
// ---- yolox ----//
bool Yolox_Pre_Processing(cv::Mat& img) {

	for (int c = 0; c < 3; ++c){
		for (int h = 0; h < img.rows; ++h){
			//获取第i行首像素指针 
			cv::Vec3b* p1 = img.ptr<cv::Vec3b>(h);
			for (int w = 0; w < img.cols; ++w){
				pred_data.intput[c * img.cols * img.rows + h * img.cols + w] = (p1[w][c]);   //yolox
			}
		}
	}
	//std::cout << "Yolox Pre PASS..." << std::endl;
	return true;
}

// --------------------------------- 后处理 --------------------------------- //
// ---- yolov5 ----//
static inline float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
	int xx1, yy1, xx2, yy2;

	xx1 = std::max(rect1.x, rect2.x);
	yy1 = std::max(rect1.y, rect2.y);
	xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
	yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

	int insection_width, insection_height;
	insection_width = std::max(0, xx2 - xx1 + 1);
	insection_height = std::max(0, yy2 - yy1 + 1);

	float insection_area, union_area, iou;
	insection_area = float(insection_width) * insection_height;
	union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
	iou = insection_area / union_area;
	return iou;
}

static bool nms_boxes() {
	BBOX bbox;
	std::vector<BBOX> bboxes;
	int i, j;
	for (i = 0; i < Process_data.boxes.size(); i++)
	{
		bbox.box = Process_data.boxes[i];
		bbox.confidence = Process_data.confidences[i];
		bbox.index = i;
		bboxes.push_back(bbox);
	}

	sort(bboxes.begin(), bboxes.end(), [](const BBOX& a, const BBOX& b) { return a.confidence > b.confidence; });
	int updated_size = bboxes.size();
	for (i = 0; i < updated_size; i++)
	{
		if (bboxes[i].confidence < cfg_info.Pred.conf)
			continue;

		Process_data.indices.push_back(bboxes[i].index);
		for (j = i + 1; j < updated_size; j++)
		{
			float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
			if (iou > cfg_info.Pred.iou)
			{
				bboxes.erase(bboxes.begin() + j);
				j--;
				updated_size = bboxes.size();
			}
		}
	}

	return true;
}

bool Yolov5_Post_Processing() {
	//清空 0ms
	Process_data.boxes.clear();
	Process_data.confidences.clear();
	Process_data.Classes.clear();
	Process_data.indices.clear();
	//1ms
	for (int i = 0; i < global_data.Output_Dim[1]; i++) {//global_data.Output_Dim = [1,29200,5+cls]

		
			float temp_conf = *std::max_element(&pred_data.output[i * (5 + Process_data.Num_Classes) + 5],
				&pred_data.output[(i + 1) * (5 + Process_data.Num_Classes)]);

			if (temp_conf < cfg_info.Pred.conf)
			{
				continue;
			}
			//xywh
			cv::Rect temp;
			temp.x = ((float*)pred_data.output)[i * (5 + Process_data.Num_Classes)];
			temp.y = ((float*)pred_data.output)[i * (5 + Process_data.Num_Classes) + 1];
			temp.width = ((float*)pred_data.output)[i * (5 + Process_data.Num_Classes) + 2];
			temp.height = ((float*)pred_data.output)[i * (5 + Process_data.Num_Classes) + 3];
			//class
			int tempClass = std::max_element(&pred_data.output[i * (5 + Process_data.Num_Classes) + 5],
				&pred_data.output[(i + 1) * (5 + Process_data.Num_Classes)])
				- &pred_data.output[i * (5 + Process_data.Num_Classes) + 5];

			Process_data.boxes.push_back(temp);
			Process_data.Classes.push_back(tempClass);
			Process_data.confidences.push_back(((float*)pred_data.output)[i * (5 + Process_data.Num_Classes) + 4] * temp_conf);

		
		
	}

	//非极大值抑制 10ms
	//if (!nms_boxes()){
	//	return false;
	//}
	//1ms
	cv::dnn::NMSBoxes(Process_data.boxes, Process_data.confidences, cfg_info.Pred.conf, cfg_info.Pred.iou, Process_data.indices);

	return true;
}

// ---- yolox ----//
static bool generate_grids_and_stride(){
	//网格落点
	for (auto stride : Process_data.strides){
		int num_grid_w = global_data.Input_Dim[2] / stride;
		int num_grid_h = global_data.Input_Dim[2] / stride;
		for (int g1 = 0; g1 < num_grid_w; g1++){
			for (int g0 = 0; g0 < num_grid_h; g0++){
				GridAndStride gs;
				gs.grid0 = g0;
				gs.grid1 = g1;
				gs.stride = stride;
				Process_data.grid_strides.push_back(gs);
			}
		}
	}
	return true;
}

bool Yolox_Post_Processing() {

	//清空
	Process_data.boxes.clear();
	Process_data.confidences.clear();
	Process_data.Classes.clear();
	Process_data.indices.clear();
	Process_data.grid_strides.clear();

	//网格落点
	generate_grids_and_stride();

	for (int anchor_idx = 0; anchor_idx < Process_data.grid_strides.size(); anchor_idx++){
		const int grid0 = Process_data.grid_strides[anchor_idx].grid0; // H
		const int grid1 = Process_data.grid_strides[anchor_idx].grid1; // W
		const int stride = Process_data.grid_strides[anchor_idx].stride; // stride+

		const int basic_pos = anchor_idx * (Process_data.Num_Classes + 5);       //[,?......] 7

		float x_center = (pred_data.output[basic_pos + 0] + grid0) * stride;
		float y_center = (pred_data.output[basic_pos + 1] + grid1) * stride;
		float w = exp(pred_data.output[basic_pos + 2]) * stride;
		float h = exp(pred_data.output[basic_pos + 3]) * stride;
		float x0 = x_center - w * 0.5f;     //除法比乘法慢
		float y0 = y_center - h * 0.5f;
		float box_objectness = pred_data.output[basic_pos + 4];

		for (int class_idx = 0; class_idx < Process_data.Num_Classes; ++class_idx)  { //类别数量
		
				float box_cls_score = pred_data.output[basic_pos + 5 + class_idx];
				float box_prob = box_objectness * box_cls_score;

				//跳过置信度低的
				if (box_prob < cfg_info.Pred.conf)
					continue;

				cv::Rect rect;
				rect.x = x_center;
				rect.y = y_center;
				rect.width = w;
				rect.height = h;

				Process_data.Classes.push_back(class_idx);
				Process_data.confidences.push_back((float)box_prob);
				Process_data.boxes.push_back(rect);

			
			

		}
	}
	//NMS
	cv::dnn::NMSBoxes(Process_data.boxes, Process_data.confidences, cfg_info.Pred.conf, cfg_info.Pred.iou, Process_data.indices);

	return true;
}

// --------------------------------- Init --------------------------------- //
static inline bool Init_Pre() {
	//设置预处理
	if (cfg_info.Pred.frame == 0) {
		global_data.Pre_process = Yolov5_Pre_Processing;
	}
	if (cfg_info.Pred.frame == 1) {
		global_data.Pre_process = Yolox_Pre_Processing;
	}
	return true;
}

static inline bool Init_Post() {
	//设置后处理
	if (cfg_info.Pred.frame == 0) {
		global_data.Post_poress = Yolov5_Post_Processing;
	}
	if (cfg_info.Pred.frame == 1) {
		global_data.Post_poress = Yolox_Post_Processing;
	}
	return true;
}

static bool Auto_select_frame() {

	int str_index = 0;
	std::string temp = cfg_info.Pred.engine_path;

	//只需要剔除一次末尾的"."
	temp = temp.substr(0, temp.find_last_of("."));

	//正向剔除所有的路径节点"/"
	while (str_index != -1){
		str_index = temp.find("/");	//第一次出现的索引
		temp = temp.substr(str_index + 1, temp.size());
	}

	//查找关键字
	if (temp.find("yolox") != -1){
		std::cout << "engine文件名找到匹配框架的字符串,将使用[ yolox ]框架" << std::endl;
		global_data.Pre_process = Yolox_Pre_Processing;
		global_data.Post_poress = Yolox_Post_Processing;
		return true;
	}

	if (temp.find("yolov5") != -1) {
		std::cout << "engine文件名找到匹配框架的字符串,将使用[ yolov5 ]框架" << std::endl;
		global_data.Pre_process = Yolov5_Pre_Processing;
		global_data.Post_poress = Yolov5_Post_Processing;
		return true;
	}

	std::cout << "engine文件名不存在匹配框架的字符串,将使用ini预设框架" << std::endl;
	return false;
}

bool Init_frame() {

	//自动选择框架
	if (Auto_select_frame()){
		return true;
	};

	//初始化预处理
	if (Init_Pre()) {
		return true;
	}

	//初始化后处理
	if (Init_Post()) {
		return true;
	}

	return true;
}
