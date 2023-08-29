#include <math.h>
#include <windows.h>

#include "msdk.h"
#include"sf-info.h"			//使用全局config_info
#include "config-info.h"	//声明结构体
#include"move-info.h"		

static struct Logitech_info Log_info;		//罗技接口
static struct KmBox_info Km_info;			//kmbox接口
static struct FeiOrYi_info FOY_info;		//飞易来/易键鼠接口
static struct Send_info Sen_info;			//Sendinput接口


// -----------------------------------   罗技移动  -----------------------------------
static inline bool Get_NT_status(PCWSTR device_name) {


	UNICODE_STRING name;
	OBJECT_ATTRIBUTES attr{};
	RtlInitUnicodeString(&name, device_name);
	InitializeObjectAttributes(&attr, &name, 0, NULL, NULL);

	//打开.Net
	Log_info.LG_status = NtCreateFile(&Log_info.LG_input, GENERIC_WRITE | SYNCHRONIZE, &attr, &Log_info.LG_io, 0,
		FILE_ATTRIBUTE_NORMAL, 0, 3, FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT, 0, 0);
	if (!(NT_SUCCESS(Log_info.LG_status))) {
		return false;
	}

	//Log_info.LG_found_mouse = TRUE;
	return true;
}

static inline BOOL callmouse(MOUSE_IO* buffer) {
	//发送
	IO_STATUS_BLOCK block;
	if (!(NtDeviceIoControlFile(Log_info.LG_input, 0, 0, 0, &block, 0x2a2010, buffer, sizeof(MOUSE_IO), 0, 0) == 0L)) {
		return false;
	}
	return true;
}

static inline bool Init_Drive() {

	if (Log_info.LG_input == 0) {
		//初始化1
		wchar_t buffer0[] = L"\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
		if (Get_NT_status(buffer0)) {
			return true;
		}
		//初始化2
		wchar_t buffer1[] = L"\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}";
		if (Get_NT_status(buffer1)) {
			return true;
		}
	}
	if (Log_info.LG_status == 0) {
		return false;
	}
	return true;
}

static bool LG_Close() {
	//释放驱动
	if (Log_info.LG_input != NULL) {
		NtClose(Log_info.LG_input);
		Log_info.LG_input = 0;
	}
	return true;
}

static inline bool Init_IO() {
	//初始化IO
	Log_info.io.unk1 = 0;
	Log_info.io.button = 0;
	Log_info.io.x = 0;
	Log_info.io.y = 0;
	Log_info.io.wheel = 0;
	return true;
}

static inline bool Send_drive(char x, char y) {
	//发送移动到驱动
	Log_info.io.x = x;
	Log_info.io.y = y;

	if (!callmouse(&Log_info.io)) {
		return false;
	}
	return true;
}

static bool LG_move(int x, int y) {

	if (abs(x) > 127 || abs(y) > 127) {
		int x_left = x; int y_left = y;

		if (abs(x) > 127) {
			Send_drive(int(x / abs(x)) * 127, 0);
			x_left = x - int(x / abs(x)) * 127;
		}
		else {
			Send_drive(int(x), 0);
			x_left = 0;
		}

		if (abs(y) > 127) {
			Send_drive(0, int(y / abs(y)) * 127);
			y_left = y - int(y / abs(y)) * 127;
		}
		else {
			Send_drive(0, int(y));
			y_left = 0;
		}

		return LG_move(x_left, y_left);
	}
	else {
		Send_drive(x, y);
	}

	Send_drive(x, y);
	return true;
}

static inline bool Log_test() {
	//移动测试
	MOUSE_IO io{};
	io.unk1 = 0;
	io.button = 0;
	io.x = 10;
	io.y = 10;
	io.wheel = 0;
	if (!callmouse(&io)) {
		std::cout << "Log_test: 打开失败,检查GHUB(要求<=2021)" << std::endl;
		return false;
	}
	return true;
}

// -----------------------------------   kmbox移动  -----------------------------------
//设置波特率
static inline bool Set_baudrate() {
	//读写缓冲区大小
	SetupComm(Km_info.com_hwnd, 1024, 1024);

	//读写超时
	COMMTIMEOUTS Out_Time{};
	Out_Time.ReadIntervalTimeout = 1000;		//读取间隔超时,单位ms
	Out_Time.ReadTotalTimeoutConstant = 5000;	//读取总超时常数
	Out_Time.ReadTotalTimeoutMultiplier = 500;	//读取总超时乘数
	Out_Time.WriteTotalTimeoutConstant = 2000;	//写入总超时常数
	Out_Time.WriteTotalTimeoutMultiplier = 500; //写入总超时乘数
	SetCommTimeouts(Km_info.com_hwnd, &Out_Time);

	//配置串口信息
	DCB device_control_info;
	GetCommState(Km_info.com_hwnd, &device_control_info);
	device_control_info.BaudRate = 115200;			//波特率
	device_control_info.ByteSize = 8;				//8位数据位
	device_control_info.StopBits = NOPARITY;		//1个停止位	
	device_control_info.Parity = NOPARITY;			//奇偶校验位
	SetCommState(Km_info.com_hwnd, &device_control_info);

	//清空
	PurgeComm(Km_info.com_hwnd, PURGE_TXCLEAR | PURGE_RXCLEAR);

	return true;
}

static inline bool Open_Comx() {

	//打开com口
	Km_info.com_hwnd = CreateFileA(
		cfg_info.Move.comx.c_str(),		//串口号
		GENERIC_READ | GENERIC_WRITE,	//支持读写
		0,								//独占方式，串口不支持共享
		NULL,							//安全属性
		OPEN_EXISTING,					//打开现有的串口文件
		0,								//0：同步方式，FILE_FLAG_OVERLAPPED：异步方式
		NULL);							//用于复制句柄，默认值为NULL，对串口而言该参数必须置为NULL
	if (Km_info.com_hwnd == INVALID_HANDLE_VALUE) {

		std::cout << "Open_Comx: 打开串口[" << cfg_info.Move.comx.c_str() << "]失败，检查硬件和串口号" << std::endl;
		return false;
	}

	//设置波特率
	if (!Set_baudrate()) {
		std::cout << "Open_Comx: 设置串口信息(波特率)失败，检查硬件占用和波特率(115200)" << std::endl;
		return false;
	}
	return true;
}

static inline bool Send_Com(const char* str) {
	//发生命令
	if (!WriteFile(Km_info.com_hwnd, str, std::strlen(str), 0, NULL)) {
		return false;
	}
	return true;
}

static bool KM_Move(int x, int y) {
	//创建移动字符串
	std::string km_send;
	km_send = "km.move(" + std::to_string(x) + "," + std::to_string(y) + ")\r\n";
	//发送
	if (!Send_Com(km_send.c_str())) {
		std::cout << "KM_Move: 移动命令发送失败,尝试重新初始化kmbox硬件" << std::endl;
		//重新初始化
		if (!Open_Comx()) {
			return false;
		}
	}
	return true;
}

static bool KM_Close() {
	//缩放句柄
	if (Km_info.com_hwnd != NULL) {
		if (!CloseHandle(Km_info.com_hwnd)) {
			std::cout << "KM_close: 释放com串口失败,错误码: " << GetLastError() << std::endl;
			return false;
		}
	}
	return true;
}

// -----------------------------------   飞易来/易键鼠移动  -----------------------------------
static bool FOY_Move(int x, int y) {
	//移动函数
	M_MoveR(FOY_info.FOY_hwnd, x, y);
	return true;
}

static bool FOY_Close() {
	//关闭接口
	if (FOY_info.FOY_hwnd) {
		M_Close(FOY_info.FOY_hwnd);
	}
	//释放句柄
	if (FOY_info.FOY_hwnd) {
		if (!CloseHandle(FOY_info.FOY_hwnd)) {
			std::cout << "FOY_Close: 释放 非易来/易键鼠 失败,报错码:" << GetLastError() << std::endl;
			return false;
		}
	}
	return true;
}

// -----------------------------------   SendInput移动  -----------------------------------
static inline bool Create_Input() {
	//创建输入结构
	Sen_info.input.type = INPUT_MOUSE;
	Sen_info.input.mi.dx = 0;
	Sen_info.input.mi.dy = 0;
	Sen_info.input.mi.mouseData = 0;
	Sen_info.input.mi.dwFlags = MOUSEEVENTF_MOVE;   //MOUSEEVENTF_ABSOLUTE 代表决对位置  MOUSEEVENTF_MOVE代表移动事件
	Sen_info.input.mi.time = 0;
	Sen_info.input.mi.dwExtraInfo = 0;

	return true;
}

static bool Send_Move(int x, int y) {
	//设置移动
	Sen_info.input.mi.dx = x;
	Sen_info.input.mi.dy = y;
	//移动
	UINT hr = SendInput(1, &Sen_info.input, sizeof(INPUT));
	if (!hr) {
		std::cout << "Send_Move: SendInput发送失败，报错码: " << GetLastError() << std::endl;
		return false;
	}
	return true;
}

static bool Send_Close() {
	//空
	return true;
}

// -----------------------------------   初始化  -----------------------------------
static inline bool Init_SendInput() {

	if (!Create_Input()) {
		return false;
	}
	global_data.Move_R = Send_Move;
	global_data.Move_free = Send_Close;
	return true;
}

static inline bool Init_FeiorYi() {
	//易键鼠/飞易来初始化
	FOY_info.FOY_hwnd = M_Open(1);
	if (FOY_info.FOY_hwnd == INVALID_HANDLE_VALUE) {
		std::cout << "Init_FeiorYi: 打开硬件失败,检查硬件" << std::endl;
		return false;
	}
	//函数地址
	global_data.Move_R = FOY_Move;
	global_data.Move_free = FOY_Close;

	return true;
}

static inline bool Init_Serail() {
	//打开com口
	if (!Open_Comx()) {
		return false;
	}
	//函数地址
	global_data.Move_R = KM_Move;
	global_data.Move_free = KM_Close;
	return true;
}

static inline bool Init_Logitech() {
	//初始化驱动
	if (!Init_Drive()) {
		std::cout << "Init_Logitech: NT初始化失败" << std::endl;
		return false;
	}
	//初始化IO
	if (!Init_IO()) {
		return false;
	}
	//测试移动
	//if (!Log_test()) {
	//	return false;
	//}

	//函数地址
	global_data.Move_R = LG_move;
	global_data.Move_free = LG_Close;
	return true;
}


bool Init_Move() {
	//初始化移动方法
	if (cfg_info.Move.move_manner == 0) {
		if (!Init_Logitech()) {
			std::cout << "Init_Move: 罗技初始化失败" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 1) {
		if (!Init_Serail()) {
			std::cout << "Init_Move: KmBox初始化失败" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 2) {
		if (!Init_FeiorYi()) {
			std::cout << "Init_Move: 飞易来/易键鼠初始化失败" << std::endl;
			return false;
		}
	}
	if (cfg_info.Move.move_manner == 3) {
		if (!Init_SendInput()) {
			std::cout << "Init_Move: SendInput初始化失败" << std::endl;
			return false;
		}
	}
	std::cout << "Mouse movement initialization PASS..." << std::endl;
	return true;
}
