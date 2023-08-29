
#include "sf-info.h"
#include "win-capture.h"
#include "config-info.h"

static struct win_capture win_data;

// ------------------------------------  获取句柄 ------------------------------------ //
static inline bool Get_Foreground_Window() {
	//获取当前顶层窗口句柄
	win_data.window_hwnd = GetForegroundWindow();
	if (win_data.window_hwnd == NULL) {
		std::cout << "Get_Foreground_Window: 获取顶层窗口失败,错误码:" << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool Get_Specified_Window() {
	//获取指定窗口句柄
	win_data.window_hwnd = FindWindowA(NULL, cfg_info.Windows.win32_name.c_str());
	if (win_data.window_hwnd == NULL) {
		std::cout << "Get_Specified_Window: 获取指定窗口失败,错误码: " << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool Win_Size_Is_Nice() {

	//截图原点小于窗口原点的,预防过小的窗口
	if ((global_data.origin_x) < 0) {
		std::cout << "Win_Screenshot_size: 当前顶层窗口不符合条件,置顶符合条件的窗口" << std::endl;
		return false;
	}

	if ((global_data.origin_y) < 0) {
		std::cout << "Win_Screenshot_size: 当前顶层窗口不符合条件,置顶符合条件的窗口" << std::endl;
		return false;
	}
	return true;
}

static inline bool Win_Screenshot_size() {

	RECT rect;
	//获取窗口信息
	if (!GetWindowRect(win_data.window_hwnd, &rect)) {
		std::cout << "Win_Screenshot_size: 获取窗口宽高失败" << std::endl;
		return false;
	}

	//窗口宽高
	global_data.window_width = rect.right - rect.left;
	global_data.window_height = rect.bottom - rect.top;

	//窗口中心点
	global_data.cx = global_data.window_width * 0.5f;
	global_data.cy = global_data.window_height * 0.5f;

	//计算截图原点
	global_data.origin_x = global_data.cx - (global_data.Input_Dim[2] * 0.5f);
	global_data.origin_y = global_data.cy - (global_data.Input_Dim[3] * 0.5f);

	//窗口是否符合截图条件
	if (!Win_Size_Is_Nice()) {
		return false;
	}

	//截图需要的字节数
	win_data.size = global_data.Input_Dim[2] * global_data.Input_Dim[3] * 4;

	return true;
}

static inline bool Set_Windows_Hwnd_and_size() {

	if (cfg_info.Windows.win32_method == 1) {
		//获取顶层窗口
		if (!Get_Foreground_Window()) {
			return false;
		}
	}
	if (cfg_info.Windows.win32_method == 0) {
		//获取指定窗口
		if (!Get_Specified_Window()) {
			return false;
		}
	}

	//设置截图范围
	if (!Win_Screenshot_size()) {
		return false;
	}

	return true;
}

static inline bool Hwnd_Not_Is_Null_And_IsWindow(HWND info) {
	//句柄不为空
	if (info == NULL) {
		return  false;
	}

	//是一个窗口句柄
	if (!IsWindow(info)) {	//不存在返回0

		return  false;
	}

	return true;
}

static inline bool Init_hwnd() {
	//获取窗口句柄，和设置截图范围
	while (!Set_Windows_Hwnd_and_size()) {
		//等待2s
		Sleep(2000);
	}

	//检查句柄
	if (!Hwnd_Not_Is_Null_And_IsWindow(win_data.window_hwnd)) {
		return false;
	}

	return true;
}

// ------------------------------------  获取DC  ------------------------------------ //
static inline bool Get_Window_DC() {
	//获取窗口DC(内存)
	win_data.Window_DC = GetWindowDC(win_data.window_hwnd);
	if (win_data.Window_DC == NULL) {
		return false;
	}
	return  true;
}

static inline bool Create_Compatible_DC() {
	//创建兼容DC
	win_data.compatible_DC = CreateCompatibleDC(win_data.Window_DC);
	if (win_data.compatible_DC == NULL) {
		return false;
	}
	return true;
}

// ------------------------------------  获取兼容位图  ------------------------------------ //
static inline bool Create_Compatible_Bitmap() {
	//创建兼容位图对象
	win_data.BitMap = CreateCompatibleBitmap(win_data.Window_DC, global_data.Input_Dim[2], global_data.Input_Dim[3]);
	if (win_data.BitMap == NULL) {
		return false;
	}
	//绑定位图到兼容DC(内存)
	SelectObject(win_data.compatible_DC, win_data.BitMap);
	return true;
}

static inline bool create_img() {
	//global_data.img.create(cv::Size(global_data.Input_Dim[2], global_data.Input_Dim[3]), CV_8UC4);
	//创建固定维度的空矩阵
	win_data.win_img = cv::Mat(global_data.Input_Dim[2], global_data.Input_Dim[3], CV_8UC4);
	return true;
}

static inline bool Create_Map() {
	// ------------ 顺序不能乱 ------------ //
	//创建兼容DC
	if (!Create_Compatible_DC()) {
		std::cout << "Create_Map: 创建兼容DC失败" << std::endl;
		return false;
	}
	//创建兼容位图
	if (!Create_Compatible_Bitmap()) {
		std::cout << "Create_Map: 创建兼容Map失败" << std::endl;
		return false;
	}
	//创建固定维度的空矩阵
	if (!create_img()) {
		std::cout << "Create_Map: 创建Mat失败" << std::endl;
		return false;
	}
	return true;
}

// ------------------------------------  Win32_capture  ------------------------------------ //
static inline bool Get_Bitmap_Bits() {
	//获取位图像素
	LONG hr = GetBitmapBits(win_data.BitMap, win_data.size, win_data.win_img.data);	//img = global_data.img
	if (hr == E_NOINTERFACE) {
		std::cout << "BitBlt_capture: 无法将示例抓取器筛选器添加到图形" << std::endl;
		return false;
	}
	if (hr == E_POINTER) {
		std::cout << "BitBlt_capture: 指针错误" << std::endl;
		return false;
	}
	if (hr == E_UNEXPECTED) {
		std::cout << "BitBlt_capture: 意外错误" << std::endl;
		return false;
	}
	return true;
}

static inline bool Window_again_should_init() {
	//窗口句柄是否为空且指向一个窗口
	if (!Hwnd_Not_Is_Null_And_IsWindow(win_data.window_hwnd)) {

		std::cout << "BitBlt_capture: 窗口丢失，重新初始化窗口" << std::endl;
		//重新初始化句柄
		if (!Init_hwnd()) {
			std::cout << "Window_again_should_init: 重新初始化句柄失败" << std::endl;
			return false;
		}
		//重新获取DC
		if (!Get_Window_DC()) {
			std::cout << "Window_again_should_init: 重新获取窗口DC失败" << std::endl;
		}
		//重新创建兼容DC
		if (!Create_Compatible_DC()) {
			std::cout << "Window_again_should_init: 重新创建兼容DC失败" << std::endl;
			return false;
		}
		//重新创建兼容位图
		if (!Create_Compatible_Bitmap()) {
			std::cout << "Window_again_should_init: 重新创建兼容Map失败" << std::endl;
			return false;
		}
	}
	return true;
}

static inline bool Check_Is_Foreground_Window() {
	// ------------ 未完成，未使用 ------------ //
	//获取当前置顶窗口
	HWND temp_hwnd = GetForegroundWindow();
	//当前置顶窗口句柄与截图窗口句柄不一样
	if (temp_hwnd != win_data.window_hwnd){
		return true;
	}
	return true;
}

static inline bool BitBlt_() {
	//翻转
	LONG hr = BitBlt(win_data.compatible_DC, 0, 0, global_data.Input_Dim[2], global_data.Input_Dim[3], win_data.Window_DC, global_data.origin_x, global_data.origin_y, SRCCOPY);
	if (FAILED(hr)) {
		std::cout << "BitBlt_capture: BitBlt失败,错误码: "<<GetLastError() << std::endl;
		return false;
	}
	return true;
}

static inline bool BitBlt_capture() {

	//检查窗口是否丢失and从新初始化
	if (!Window_again_should_init()) {
		std::cout << "BitBlt_capture: 重新初始化窗口失败" << std::endl;
		return false;
	}

	//检查是否置顶窗口
	//if (!Check_Is_Foreground_Window()){
	//}

	//翻转窗口图片到兼容DC
	if (!BitBlt_()){
		std::cout << "BitBlt_capture: BitBlt失败" << std::endl;
		return false;
	}
	//获取位图像素
	if (!Get_Bitmap_Bits()) {
		std::cout << "BitBlt_capture: Get_Bitmap_Bits失败" << std::endl;
		return false;
	}
	//4-> 顺带将Get_Bitmap_Bits后的位图放入全局图片中
	(win_data.win_img, global_data.img, cv::COLOR_BGRA2BGR);

	return true;
}

// ------------------------------------  Win32_Free  ------------------------------------ //
bool Free_BitBlt() {

	if (win_data.Window_DC) {
		DeleteDC(win_data.Window_DC);
	}
	if (win_data.compatible_DC) {
		DeleteDC(win_data.compatible_DC);
	}
	if (win_data.BitMap) {
		DeleteObject(win_data.BitMap);
	}
	if (win_data.window_hwnd) {
		CloseHandle(win_data.window_hwnd);
	}

	return true;
}

// ------------------------------------  Win32_Init  ------------------------------------ //
bool Init_BitBlt() {

	//获取窗口句柄，并设置截取范围
	if (!Init_hwnd()) {
		std::cout << "Init_BitBlt: 初始化句柄失败" << std::endl;
		return false;
	}
	//获取窗口DC
	if (!Get_Window_DC()) {
		std::cout << "Init_BitBlt: 获取窗口DC失败" << std::endl;
	}
	//创建Map资源
	if (!Create_Map()) {
		std::cout << "Init_BitBlt: 初始化位图失败" << std::endl;
		return false;
	}

	global_data.capture_map = BitBlt_capture;
	global_data.capture_free = Free_BitBlt;

	return true;
}
