
#include <cmath>

#include "sf-info.h"
#include "aim-info.h"
#include "config-info.h"
#include "process-info.h"
#include <chrono>

#define pi 3.14159265358979323846   

static struct PID pid_x;		
static struct PID pid_y;
int kg;




// ----------------------------- FOV ----------------------------- //
static inline void FOV_X(float& move_x) {
	float x = (global_data.window_width * 0.5f) / tan((cfg_info.Fov.game_HFOV * pi) / 180 * 0.5f);
	move_x = atan(move_x / x) * (cfg_info.Fov.game_x_pixel / (2 * pi));
}

static inline void FOV_Y(float& move_y) {
	float y = (global_data.window_height * 0.5f) / tan((cfg_info.Fov.game_VFOV * pi) / 180 * 0.5f);
	move_y = atan(move_y / y) * (cfg_info.Fov.game_y_pixel / pi);
}

// ----------------------------- PID ----------------------------- //
//增量式
static inline void Incremental(PID pid, float& move, const float& kp, const float& ki, const float& kd, const int& num) {
	pid.SetSpeed = move;					//记录本次移动距离
	pid.Error = pid.SetSpeed - pid.ActualSpeed;		//记录本次与上一次的误差
	pid.IncrementSpeed = kp * (pid.Error - pid.Err_Next);// +(ki * pid.Error) + kd * (pid.Error - 2 * pid.Err_Next + pid.Err_last);

	//简易抗积分饱和
	pid.Sample_frame++;
	if (pid.Sample_frame % num == 0) {
		pid.ActualSpeed += pid.IncrementSpeed;	//积分累积
		pid.Sample_frame = 0;
	}

	//pid.ActualSpeed += pid.IncrementSpeed;	//积分累积
	pid.Err_last = pid.Err_Next;			//
	pid.Err_Next = pid.Error;				//当前误差转为下一次的参照
	move = pid.IncrementSpeed;
}

//x轴
static inline void PID_X(float& move_x) {
	Incremental(pid_x, move_x, cfg_info.Pid.kp_x, cfg_info.Pid.ki_x, cfg_info.Pid.kd_x, cfg_info.Pid.sample_num);
}

//y轴
static inline void PID_Y(float& move_y) {
	Incremental(pid_y, move_y, cfg_info.Pid.kp_y, cfg_info.Pid.ki_y, cfg_info.Pid.kd_y, cfg_info.Pid.sample_num);
}

//偏移
static inline void Offset(int &y,int &width) {


	if ((0.f< cfg_info.Aim.move_offset) && (cfg_info.Aim.move_offset <= 1.f)) {
		y = y - (width * cfg_info.Aim.move_offset);
		return;
	}
	
	//限位
	if (cfg_info.Aim.move_offset>1.f)
		y = y - (width * 1.f);
	else 
		y = y - (width * 0.1f);
}


// ----------------------------- 计算分支 ----------------------------- //
//自瞄范围
static inline bool Check_Range(float& move_x, float& move_y) {
	if (cfg_info.Aim.range_top < abs(move_y) || cfg_info.Aim.range_bottom < abs(move_y) ||
		cfg_info.Aim.range_left < abs(move_x) || cfg_info.Aim.range_right < abs(move_x)
		) {

		return false;
	}
	return true;
}

static inline bool FOV_PID(float& move_x, float& move_y) {

	//FOV计算
	if (cfg_info.Fov.fov_off != 0) {
		FOV_X(move_x);
		FOV_Y(move_y);
	}

	//PID计算
	if (cfg_info.Pid.pid_off != 0) {
		PID_X(move_x);
		PID_Y(move_y);
	}

	return true;
}

// ----------------------------- 自动扳机 ----------------------------- //
//自动扳机单击
static inline void Key_click() {

	//按下
	INPUT input{};
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;   //MOUSEEVENTF_LEFTDOWN 左键按下
	input.mi.time = 0;
	input.mi.dwExtraInfo = 0;
	SendInput(1, &input, sizeof(INPUT));

	//松开
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTUP;   // MOUSEEVENTF_LEFTUP  左键松开
	input.mi.time = 0;
	input.mi.dwExtraInfo = 0;

	SendInput(1, &input, sizeof(INPUT));
}

//扳机延时
std::chrono::system_clock::time_point Sleep_Start = std::chrono::system_clock::now();
static inline bool Auto_Sleep() {
	std::chrono::system_clock::time_point Sleep_End = std::chrono::system_clock::now();
	return std::chrono::duration_cast <std::chrono::milliseconds> (Sleep_End - Sleep_Start).count() >= cfg_info.Fire.auto_sleep;
}

//自动扳机
static inline void Auto_fire_func(float* move_x, float* move_y) {
	//自动扳机方式 - 移动后开枪
	if (cfg_info.Fire.auto_method  == 0) {
		global_data.Move_R(int(*move_x), int(*move_y));
		if (Auto_Sleep()) {
			Key_click();
			Sleep_Start = std::chrono::system_clock::now();
		}
	}

	if (cfg_info.Fire.auto_method  == 1) {
		//范围开枪
		if ((Process_data.target_xywh.width * 0.5f) > abs(*move_x) && (Process_data.target_xywh.height * 0.5f) > abs(*move_y) && Auto_Sleep()) {
			Key_click();
			Sleep_Start = std::chrono::system_clock::now();
		}
	}
}
// ----------------------------- 移动逻辑 ----------------------------- //
//移动逻辑
static inline bool Move_logic(float* move_x, float* move_y) {

	//自瞄方式
	if (cfg_info.Key.key_method == 0) {
		//按键移动
		if (GetAsyncKeyState(cfg_info.Key.button_key1) || GetAsyncKeyState(cfg_info.Key.button_key2)) {
			//自动扳机
			if (cfg_info.Fire.auto_off == 1) {
				Auto_fire_func(move_x, move_y);

			}
			else {
				global_data.Move_R(int(*move_x), int(*move_y));
			}
		}
	}
	else {
		if (cfg_info.Fire.auto_off == 1) {
			Auto_fire_func(move_x, move_y);
		}
		else {
			global_data.Move_R(int(*move_x), int(*move_y));
		}
	}

	return true;
}

// ----------------------------- aim Loop ----------------------------- //
DWORD WINAPI aim_func() {


	while (true) {
		//等待启动信号
		WaitForSingleObject(Aim_Event, INFINITE);
		//程序是否退出
		if (Thread_Should_Stop()) {
			break;
		}
		//清空
		Process_data.recent.clear();
		if (cfg_info.Aim.cla_off) {


			for (int i = 0; i < Process_data.indices.size(); i++) {
				if (Process_data.Classes[Process_data.indices[i]] == cfg_info.Aim.label_chose) {
					for (int i = 0; i < Process_data.indices.size(); i++) {


						Process_data.recent.push_back(
							pow((global_data.origin_x + Process_data.boxes[Process_data.indices[i]].x) - global_data.cx, 2) +
							pow((global_data.origin_y + Process_data.boxes[Process_data.indices[i]].y) - global_data.cy, 2));


					}

					//最近目标的索引
					int idx = Process_data.indices[
						std::distance(std::begin(Process_data.recent),

							std::min_element(std::begin(Process_data.recent), std::end(Process_data.recent))//min_element()
						)	//distance()
					];	//Process_data.indices[]

					//最近目标的x	(在图片坐标)
					Process_data.target_xywh.x = Process_data.boxes[idx].x;
					//最近目标的y	(在图片坐标)
					Process_data.target_xywh.y = Process_data.boxes[idx].y;
					//最近目标的width	(在图片坐标)
					Process_data.target_xywh.width = Process_data.boxes[idx].width;
					//最近目标的height	(在图片坐标)
					Process_data.target_xywh.height = Process_data.boxes[idx].height;

					//Debug 测试
					//std::cout
					//	<< "x: " << Process_data.target_xywh.x
					//	<< "\ty: " << Process_data.target_xywh.y
					//	<< "\tweidth: " << Process_data.target_xywh.width
					//	<< "\theight: " << Process_data.target_xywh.height
					//	<< std::endl;

					//偏移计算
					if (cfg_info.Aim.move_offset != 0) {
						Offset(Process_data.target_xywh.y, Process_data.target_xywh.width);
					}

					// 截图的原点(global_data.origin_x) + 目标中心点(Process_data.target_xywh.x) = 目标在屏幕的中心点 
					// 目标在屏幕的中心点 - 窗口中心点(global_data.cx) = 准星和目标之间的距离(move_x)
					float move_x = global_data.origin_x + Process_data.target_xywh.x - global_data.cx;
					float move_y = global_data.origin_y + Process_data.target_xywh.y - global_data.cy;

					//检查是否在自瞄范围内
					if (!Check_Range(move_x, move_y)) {
						continue;
					}

					//FOV + PID
					if (!FOV_PID(move_x, move_y)) {
						std::cout << "fov + pid计算出现未知错误，跳过本次移动" << std::endl;
						continue;
					}

					//移动逻辑
					if (!Move_logic(&move_x, &move_y)) {
						std::cout << "移动逻辑出现未知错误，跳过本次移动" << std::endl;
						continue;
					}

					
				}
				
			}
		}
		else {


			//计算所有目标的欧式距离
			for (int i = 0; i < Process_data.indices.size(); i++) {


				Process_data.recent.push_back(
					pow((global_data.origin_x + Process_data.boxes[Process_data.indices[i]].x) - global_data.cx, 2) +
					pow((global_data.origin_y + Process_data.boxes[Process_data.indices[i]].y) - global_data.cy, 2));


			}

			//最近目标的索引
			int idx = Process_data.indices[
				std::distance(std::begin(Process_data.recent),

					std::min_element(std::begin(Process_data.recent), std::end(Process_data.recent))//min_element()
				)	//distance()
			];	//Process_data.indices[]

			//最近目标的x	(在图片坐标)
			Process_data.target_xywh.x = Process_data.boxes[idx].x;
			//最近目标的y	(在图片坐标)
			Process_data.target_xywh.y = Process_data.boxes[idx].y;
			//最近目标的width	(在图片坐标)
			Process_data.target_xywh.width = Process_data.boxes[idx].width;
			//最近目标的height	(在图片坐标)
			Process_data.target_xywh.height = Process_data.boxes[idx].height;

			//Debug 测试
			//std::cout
			//	<< "x: " << Process_data.target_xywh.x
			//	<< "\ty: " << Process_data.target_xywh.y
			//	<< "\tweidth: " << Process_data.target_xywh.width
			//	<< "\theight: " << Process_data.target_xywh.height
			//	<< std::endl;

			//偏移计算
			if (cfg_info.Aim.move_offset != 0) {
				Offset(Process_data.target_xywh.y, Process_data.target_xywh.width);
			}

			// 截图的原点(global_data.origin_x) + 目标中心点(Process_data.target_xywh.x) = 目标在屏幕的中心点 
			// 目标在屏幕的中心点 - 窗口中心点(global_data.cx) = 准星和目标之间的距离(move_x)
			float move_x = global_data.origin_x + Process_data.target_xywh.x - global_data.cx;
			float move_y = global_data.origin_y + Process_data.target_xywh.y - global_data.cy;

			//检查是否在自瞄范围内
			if (!Check_Range(move_x, move_y)) {
				continue;
			}

			//FOV + PID
			if (!FOV_PID(move_x, move_y)) {
				std::cout << "fov + pid计算出现未知错误，跳过本次移动" << std::endl;
				continue;
			}

			//移动逻辑
			if (!Move_logic(&move_x, &move_y)) {
				std::cout << "移动逻辑出现未知错误，跳过本次移动" << std::endl;
				continue;
			}
		}
		
	}
	std::cout << "自瞄线程退出..." << std::endl;
	return true;
}

