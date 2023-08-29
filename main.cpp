#include <windows.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>

#include "sf-info.h"
#include "config-info.h"
#include "create-resource.h"
#include "process-info.h"
#include "dxgi-capture.h"
#include "win-capture.h"
#include "move-info.h"
#include "aim-info.h"
#include <ort-gpu.h>
#include <numeric>

#define MUTEX_REPEAT L"Repeat"						//互斥体，用于重复运行检测
#define WINDOWS_SHOW_NAME "Yolo_V5"		//
#define EVENT_CAPTURE_RESTART1 L"Aim_Event"			//自瞄事件


//全局结构体
struct Config_Info cfg_info;		//config参数
struct global_info global_data;		//全局信息
struct Porocess_info Process_data;	//后处理信息,目标信息

//当前全局结构体
static struct Time_info Time_i;	//后处理信息,目标信息

//互斥体
static HANDLE repeat_run = nullptr;		//互斥体,重复检测

//事件
HANDLE Aim_Event = nullptr;		//控制自瞄，全局事件

//线程句柄
static HANDLE aim_move_thread = nullptr;		//自瞄线程
static HANDLE Dynamic_Read_thread = nullptr;		//动态线程


//当前全局
static MSG msg;
static float Version = 5.0;					//版本标识
static bool Init_Ready = FALSE;				//初始化完成
static bool Set_Console_Ready = FALSE;		//设置控制台回调函数
static bool While_Should_Stop = FALSE;      //线程停止标识

//开关标识
static int key_state = 0;					//按键状态
static int Aim_state = FALSE;				//开关状态

//函数声明
static void Free();
static void Exit_Coda();
static inline void close_handle(HANDLE* handle);

// -------------------------------  Tool  ------------------------------- //

//全局函数，向外提供 线程停止信号
bool Thread_Should_Stop() {
	return !!While_Should_Stop;
}

//所有初始化就绪，开始信号
static inline bool Start_Ready() {
	return !!Set_Console_Ready && !!Init_Ready;
}

//获取开关状态
static inline bool get_switch_state() {
	return !!Aim_state;
}

//画框函数
static inline bool Draw_box() {
	

	if (cfg_info.Aim.cla_off) {


		for (int i = 0; i < Process_data.indices.size(); i++) {
			if (Process_data.Classes[Process_data.indices[i]] == cfg_info.Aim.label_chose) {



				BBOX box;
				cv::rectangle(global_data.img,
					cv::Rect(
							Process_data.boxes[Process_data.indices[i]].x - Process_data.boxes[Process_data.indices[i]].width * 0.5f,
							Process_data.boxes[Process_data.indices[i]].y - Process_data.boxes[Process_data.indices[i]].height * 0.5f,
							Process_data.boxes[Process_data.indices[i]].width,
							Process_data.boxes[Process_data.indices[i]].height),
						cv::Scalar(0, 255, 0), 2, 8, 0);
			}
		}
	}
	else {
		for (int i = 0; i < Process_data.indices.size(); i++) {


				BBOX box;
				cv::rectangle(global_data.img,
					cv::Rect(
						Process_data.boxes[Process_data.indices[i]].x - Process_data.boxes[Process_data.indices[i]].width * 0.5f,
						Process_data.boxes[Process_data.indices[i]].y - Process_data.boxes[Process_data.indices[i]].height * 0.5f,
						Process_data.boxes[Process_data.indices[i]].width,
						Process_data.boxes[Process_data.indices[i]].height),
					cv::Scalar(0, 255, 0), 2, 8, 0);
			
		}

	}
		
			
		
		

		
	
	cv::imshow(WINDOWS_SHOW_NAME, global_data.img);
	cv::waitKey(1);
	return true;
}

static inline bool Win_Show_Switch() {

	if (!cfg_info.Windows.show) {
		//如果已经有一个窗口存在，则注销
		if (cv::getWindowProperty(WINDOWS_SHOW_NAME, cv::WND_PROP_VISIBLE)) {
			cv::destroyWindow(WINDOWS_SHOW_NAME);
		}
		return false;
	}
	//显示窗口
	Draw_box();
	return true;
}

//打印线程计时器
std::chrono::system_clock::time_point Print_Sleep_Start = std::chrono::system_clock::now();
static inline bool print_sleep() {
	std::chrono::system_clock::time_point Print_Sleep_End = std::chrono::system_clock::now();
	return std::chrono::duration_cast <std::chrono::milliseconds> (Print_Sleep_End - Print_Sleep_Start).count() >= cfg_info.Other.console_refresh;
}

static inline void Print_info() {
	//打印输出
	if (print_sleep() && cfg_info.Other.console_refresh) {
		std::cout
			<< "循环: " << std::chrono::duration_cast<std::chrono::milliseconds>(Time_i.end - Time_i.start).count() << "ms\t"
			<< "截图: " << std::chrono::duration_cast<std::chrono::milliseconds>(Time_i.capture_end - Time_i.capture_start).count() << "ms\t"
			<< "推理: " << std::chrono::duration_cast<std::chrono::milliseconds>(Time_i.Inference_end - Time_i.Inference_start).count() << "ms\t"
			<< "主动睡眠: " << cfg_info.Pred.sleep << "ms\t"
			<< "FPS:" << 1000 / (std::chrono::duration_cast<std::chrono::milliseconds>(Time_i.end - Time_i.start).count())
			<< std::endl;

		//更新计时器
		Print_Sleep_Start = std::chrono::system_clock::now();
	}
}

//检查显卡
static bool Check_GPU() {

	int deviceCount;
	cudaDeviceProp deviceProp;
	cudaError_t cudaError;
	cudaError = cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cout << "Check_Head: 未找到CUDA设备,检查当前主机是否存在英伟达显卡。" << std::endl;
		std::cout << "-------------------------------------------------------------" << std::endl;
		return false;
		system("pause");
	}
	//获取CUDA信息
	for (int i = 0; i < deviceCount; i++) {
		cudaError = cudaGetDeviceProperties(&deviceProp, i);
		std::cout << "-------------------------------------------------------------" << std::endl;
		std::cout << "GPU型号: " << deviceProp.name << "  " << deviceProp.totalGlobalMem / 1024 / 1024 << " MB " << std::endl;
		std::cout << "-------------------------------------------------------------" << std::endl;
	}

	return true;
}

static inline bool Check_Config_Version() {
	//检查ini文件和exe版本
	if (cfg_info.Check.checking_Version != Version) {
		std::cout << "Check_Head: config与SF_trt版本不匹配,config版本:" << cfg_info.Check.checking_Version << std::endl;
		system("pause");
		return false;
	}
	return true;
}

// -------------------------------  Print_Head  ------------------------------- //
static bool Print_Head() {
	//自定义输出头
	std::stringstream sstr;
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_INTENSITY | FOREGROUND_BLUE);//设置蓝色
	std::cout << "================================================================================ " << std::endl;
	std::cout << "******************************《 Yolo_V5_Ai自瞄 》****************************** " << std::endl;
	std::cout << std::endl;
	sstr << "****************************《 Current Version: " << Version << " 》****************************";
	std::cout << sstr.str() << std::endl;
	std::cout << "================================================================================ " << std::endl;
	return true;
}

// -------------------------------  SF_TRT_Init  ------------------------------- //
static inline bool Init_capture() {
	//DXGI捕获方式
	if (cfg_info.Windows.capture_method == 0) {
		if (!Init_dxgi()) {
			Free_dxgi();
			return false;
		}
		std::cout << "Dxgi screenshot initialization PASS..." << std::endl;
	}
	//GDI捕获方式
	if (cfg_info.Windows.capture_method == 1) {
		if (!Init_BitBlt()) {
			Free_BitBlt();
			return false;
		}
		std::cout << "BitBlt screenshot initialization PASS..." << std::endl;
	}
	return true;
}

//初始化SF_trt推理需要的资源
static inline bool SF_TRT_Init() {
	//初始化cuda
	if (!ort_dml_init()) {
		return false;
	}
	//初始化截图
	if (!Init_capture()) {
		return false;
	}
	//初始化推理框架
	if (!Init_frame()) {
		return false;
	}
	else {
		std::cout << "Frame initialization PASS..." << std::endl;
	}
	//初始化移动
	if (!Init_Move()) {
		return false;
	}

	return true;
}

// -------------------------------  Init_Event  ------------------------------- //
//创建事件，返回句柄
static inline HANDLE create_event(const wchar_t* name, DWORD pid){
	HANDLE handle = nullptr;
	wchar_t new_name[64]{};

	//确保内核不会出现相同的事件
	_snwprintf(new_name, 64, L"%s%lu", name, pid);
	handle = CreateEventW(NULL, false, false, name);

	if (!handle) {
		return false;
	}
	return handle;
}

//初始化事件
static inline bool Init_Event() {
	DWORD pid = GetCurrentProcessId();

	//创事件
	Aim_Event = create_event(EVENT_CAPTURE_RESTART1, pid);
	if (!Aim_Event) {
		std::cout << "创建Aim_Event事件失败,错误码: " << GetLastError() << std::endl;
		return false;
	}
	return true;
}

// -------------------------------  Init_thread  ------------------------------- //
//初始化线程
static inline bool Init_thread() {
	//创建自瞄线程，挂起
	aim_move_thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)aim_func, 0, CREATE_SUSPENDED, 0);
	if (aim_move_thread == NULL) {
		std::cout << "Init_thread; 创建自瞄线程失败,错误码: " << GetLastError() << std::endl;
		return false;
	}
	//创建动态更新线程，挂起
	Dynamic_Read_thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)dynamic_thread_loop, 0, CREATE_SUSPENDED, 0);
	if (Dynamic_Read_thread == NULL) {
		std::cout << "Init_thread; 创建动态读取线程失败,错误码: " << GetLastError() << std::endl;
		return false;
	}	

	return true;
}

// -------------------------------  Hook 键盘 ------------------------------- //
HHOOK switch_hook;
LRESULT CALLBACK KeyboardProc(int code, WPARAM wparam, LPARAM iparam) {	//iparam：VK键值
	if (code == HC_ACTION){
		//退出按键
		if (GetAsyncKeyState(cfg_info.Key.end_key)) {
			While_Should_Stop = TRUE;
			Exit_Coda();
		}
		//开关
		if (GetAsyncKeyState(cfg_info.Key.aim_off) && key_state == 0) {
			key_state = 1;
			Aim_state = TRUE;
			std::cout << "Aim: 开" << std::endl;
		}
		if (!GetAsyncKeyState(cfg_info.Key.aim_off) && key_state == 1) {
			key_state = 2;
		}
		if (GetAsyncKeyState(cfg_info.Key.aim_off) && key_state == 2) {
			key_state = 3;
			Aim_state = FALSE;
			std::cout << "Aim: 关" << std::endl;
		}
		if (!GetAsyncKeyState(cfg_info.Key.aim_off) && key_state == 3) {
			key_state = 0;
		}
	}
	return CallNextHookEx(switch_hook, code, wparam, iparam);
}

static inline bool Init_keyboard_hook() {

	//hook全局键盘消息
	switch_hook  = SetWindowsHookEx(WH_KEYBOARD_LL, KeyboardProc, NULL, 0);

	if (!switch_hook) {
		std::cout << "hook按键失败,错误码:" << GetLastError() << std::endl;
		return false;
	}
	return true;
}

// -------------------------------  Init  ------------------------------- //
//所有初始化的开始函数
bool Init() {
	//打印自定义输出信息	
	if (!Print_Head()) {
		return false;
	}
	//检查GPU信息
	if (!Check_GPU()) {
		return false;
	}
	// 初始化ini参数	
	// Init_config()必须在Check_Config_Version()前运行
	if (!Init_config()) {
		return false;
	}
	//检查config版本是否对应
	if (!Check_Config_Version()) {
		return false;
	}
	//初始化应用资源
	if (!SF_TRT_Init()) {
		return false;
	}
	//初始化事件
	if (!Init_Event()) {
		return false;
	}
	//初始化线程
	if (!Init_thread()) {
		return false;
	}
	//hook键盘
	if (!Init_keyboard_hook()) {
		return false;
	}
	return true;
}

// -------------------------------  start_main  ------------------------------- //

static inline bool start_thread() {
	//启动自瞄线程
	DWORD hr = ResumeThread(aim_move_thread);
	if (hr == (DWORD)-1) {
		std::cout << "start_thread: 启动自瞄线程失败,错误码: " << GetLastError() << std::endl;
		return false;
	}
	//启动动态线程
	hr = ResumeThread(Dynamic_Read_thread);
	if (hr == (DWORD)-1) {
		std::cout << "start_thread: 启动动态读取线程失败,错误码: " << GetLastError() << std::endl;
		return false;
	}

	return true;
}

////推理
//static inline bool Inference() {
//	CUDA_CHECK(cudaMemcpyAsync(pred_data.buffers_ptr[0], pred_data.intput, pred_data.buffer_size[0], cudaMemcpyHostToDevice, trt.stream));
//	trt.context->executeV2(pred_data.buffers_ptr);		//同步
//	CUDA_CHECK(cudaMemcpyAsync(pred_data.output, pred_data.buffers_ptr[trt.engine->getNbBindings() - 1], pred_data.buffer_size[trt.engine->getNbBindings() - 1], cudaMemcpyDeviceToHost, trt.stream));
//	return true;
//}

bool Inference() {

	//// 创建 OrtRunOptions 结构体对象
	//OrtRunOptions* run_options;
	//g_ort->CreateRunOptions(&run_options);
	//g_ort->RunOptionsSetRunLogVerbosityLevel(run_options, ORT_LOGGING_LEVEL_WARNING);
	//g_ort->RunOptionsSetRunTag(run_options, "My_Test_Run_Tag");

	//OrtStatus* status = g_ort->Run(session, input_names.data(), &input_tensors, 1, output_names.data(), &output_tensors, 1, run_options);


	//// 释放 OrtRunOptions 对象
	//g_ort->ReleaseRunOptions(run_options);


	//if (status != nullptr) {
	//	std::cerr << "推理时发生错误: " << g_ort->GetErrorMessage(status) << std::endl;
	//	g_ort->ReleaseStatus(status);
	//	return false;
	//}

	// 将输出张量数据复制到 pred_data.output
	size_t output_data_size = std::accumulate(pred_data.op_Dim.begin(), pred_data.op_Dim.end(), 1, std::multiplies<int64_t>());
	float* output_tensor_data = nullptr;
	g_ort->GetTensorMutableData(output_tensors, reinterpret_cast<void**>(&output_tensor_data));

	// 为 pred_data.output 分配内存
	if (pred_data.output != nullptr) {
		delete[] pred_data.output;
	}
	pred_data.output = new float[output_data_size];

	// 将数据从输出张量复制到 pred_data.output
	memcpy(pred_data.output, output_tensor_data, output_data_size * sizeof(float));

	return true;
}

static inline bool hook_msg() {
	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
		if (GetMessage(&msg, NULL, 0, 0)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
	return true;
}



//主函数循环
static bool start_main() {
	//线程启动
	if (!start_thread()) {
		return false;
	}

	std::cout << "Run..." << std::endl;
	//start
	while (true) {
		//派发键盘消息
		hook_msg();

		//是否退出程序
		//必须在派发消息之后停止循环
		if (Thread_Should_Stop()){
			break;
		}


		Time_i.start = std::chrono::system_clock::now();					//循环开始时间

		//截图1-2-3ms
		Time_i.capture_start = std::chrono::system_clock::now();    		//捕获开始时间
		if (!global_data.capture_map()) {
			std::cout << "start_main: 截图发生错误，跳过本次循环" << std::endl;
			continue;
		}
		Time_i.capture_end = std::chrono::system_clock::now();		//捕获结束时间

		//预处理 2ms
		if (!global_data.Pre_process(global_data.img)) {
			std::cout << "start_main: 预处理发生错误，跳过本次循环" << std::endl;
			continue;
		}

		
		//推理 1050 yolov5 : 15ms yolox: 6ms
		Time_i.Inference_start = std::chrono::system_clock::now();     //推理开始时间
		if (!Inference()) {
			std::cout << "start_main: 推理发生错误，跳过本次循环" << std::endl;
			continue;
		}
		Time_i.Inference_end = std::chrono::system_clock::now();     //推理结束时间

		//后处理  0ms
		if (!global_data.Post_poress()) {
			std::cout << "start_main: 后处理发生错误，跳过本次循环" << std::endl;
			continue;
		}

		//自瞄打开 && 存在目标
		if (get_switch_state() && Process_data.indices.size()) 
			SetEvent(Aim_Event);

		//主动睡眠
		Sleep(cfg_info.Pred.sleep);
		Time_i.end = std::chrono::system_clock::now();//循环结束时间

		//画框
		Win_Show_Switch();
		//打印信息
		Print_info();
	}
	std::cout << "主线程结束" << std::endl;
	return true;
}

// -------------------------------  Main  ------------------------------- //
//控制台消息回调函数
BOOL WINAPI ConsoleHandler(DWORD CEvent) {
	if (CEvent == CTRL_CLOSE_EVENT) {	//消息 = 点击控制台的×
		//标识停止循环
		While_Should_Stop = TRUE;
		//退出代码
		Exit_Coda();
	}
	return true;
}

//设置控制台关闭消息
static inline bool Set_Console() {
	//设置控制台消息回调函数
	if (SetConsoleCtrlHandler((PHANDLER_ROUTINE)ConsoleHandler, TRUE) == 0) {
		std::cout << "设置控制台回调函数错误" << std::endl;
		return false;
	}
	return true;
}

//检查程序重复运行
static inline bool Check_repeat() {

	//设置最后错误为空
	SetLastError(NO_ERROR);
	//创建互斥体
	repeat_run = CreateMutexW(NULL, false, MUTEX_REPEAT);
	if (repeat_run == NULL) {
		std::cout << "创建互斥体失败" << std::endl;
		return false;
	}
	//若已存在一个互斥体，返回ERROR_ALREADY_EXISTS错误
	if (GetLastError() == ERROR_ALREADY_EXISTS) {
		close_handle(&repeat_run);
		MessageBoxA(NULL, "程序已经运行", "Yolo_V5", MB_OK);
		return false;
	}
	return true;
}

static inline bool Load_UAC() {

	BOOL retn;
	HANDLE hToken;
	LUID Luid;

	//获取打开进程的令牌
	retn = OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES, &hToken);
	if (retn != TRUE) {
		std::cout << "获取令牌句柄失败" << std::endl;
		return false;
	}

	//查找特权值
	retn = LookupPrivilegeValue(NULL, SE_DEBUG_NAME, &Luid);
	if (retn != TRUE) {
		std::cout << "获取Luid失败" << std::endl;
		return false;
	}

	//给TP和TP里的LUID结构体赋值
	TOKEN_PRIVILEGES tp{}; //新特权结构体
	tp.PrivilegeCount = 1;
	tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
	tp.Privileges[0].Luid = Luid;

	//调整权限
	AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(TOKEN_PRIVILEGES), NULL, NULL);
	if (GetLastError() != ERROR_SUCCESS) {
		std::cout << "获取UAC权限不完全或失败,错误码:"<< GetLastError() << std::endl;
		system("pause");
		return false;
	}
	else {
		std::cout << "获取UAC权限成功" << std::endl;
	}
	return true;
}

extern "C" API int main() {

	//检测程序重复运行 && 加载UAC权限
	if (Check_repeat() && Load_UAC()) {

		//设置控制台回调消息
		if (Set_Console()) 
			Set_Console_Ready = TRUE;	//设置控制台就绪标识
	
		//初始化
		if (Init()) 
			Init_Ready = TRUE;		//标志所有初始化就绪
		else
			system("pause");
		
		//运行函数
		if (Start_Ready())
			start_main();
	}

	return 0;
}

// -------------------------------  Free  ------------------------------- //
static inline void close_handle(HANDLE* handle)
{
	if (*handle) {
		CloseHandle(*handle);
		*handle = nullptr;
	}
}

//释放所有资源
static inline void Free() {
	//释放截图资源
	global_data.capture_free();

	////CUDA资源
	//Free_CUDA();

	//移动句柄释放
	global_data.Move_free();

	//释放线程句柄
	close_handle(&aim_move_thread);
	close_handle(&Dynamic_Read_thread);


	//释放可等候句柄
	close_handle(&Aim_Event);

	//释放互斥体句柄
	close_handle(&repeat_run);

	//卸载hook
	UnhookWindowsHookEx(switch_hook);
}

static void Exit_Coda() {

	SetEvent(Aim_Event);

	WaitForSingleObject(aim_move_thread, INFINITE);	//等待自瞄线程退出

	WaitForSingleObject(Dynamic_Read_thread, INFINITE);	//等待自瞄线程退出

	//PostQuitMessage(0);
	//释放函数
	Free();

}
