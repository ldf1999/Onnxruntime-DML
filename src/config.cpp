#include <iostream>
#include <string>
#include <direct.h>
#include <windows.h>
#include <fstream>

#include"sf-info.h"
#include"config-info.h"

static bool Parameter_Error = FALSE;		//参数错误标识
static bool Print_Parameter_Debug = FALSE;	//Debug打印参数控制开关 FALSE / TRUE


// ----------------------------------  tool  ---------------------------------- //
static bool Is_File(const std::string& file_path) {
	std::ifstream file(file_path.c_str());
	return file.good();
}


// ----------------------------------  Check_ini_path  ---------------------------------- //
static inline bool Check_ini() {
	if (!Is_File("config.ini")) {
		return false;
	}
	return true;
}

// ----------------------------------  设置路径  ---------------------------------- //
static inline bool Set_Config_Path() {
	//获取当前工作路径
	std::string Root_Path = getcwd(NULL, 0);
	std::string Ini_Name = "\\config.ini";
	cfg_info.Ini_Path = Root_Path + Ini_Name;
	return true;
}

// ----------------------------------  字符编码转换  ---------------------------------- //

std::wstring UTF8ToUnicode(const char* strSrc)
{
	std::wstring wstrRet;

	if (NULL != strSrc) {
		int len = MultiByteToWideChar(CP_UTF8, 0, strSrc, -1, NULL, 0) * sizeof(WCHAR);
		WCHAR* strDst = new(std::nothrow) WCHAR[len + 1];

		if (NULL != strDst) {
			MultiByteToWideChar(CP_UTF8, 0, strSrc, -1, strDst, len);
			wstrRet = strDst;;
			delete[]strDst;
		}
	}
	return wstrRet;
}

std::string UnicodeToAnsi(const WCHAR* strSrc) {
	std::string strRet;

	if (NULL != strSrc) {
		int len = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
		char* strDst = new(std::nothrow) char[len + 1];

		if (NULL != strDst) {
			WideCharToMultiByte(CP_ACP, 0, strSrc, -1, strDst, len, NULL, NULL);
			strRet = strDst;
			delete[]strDst;
		}
	}
	return strRet;
}

std::string UTF8ToAnsi(const char* strSrc) {
	return UnicodeToAnsi(UTF8ToUnicode(strSrc).c_str());
}

// ----------------------------------  检查/打印ini参数  ---------------------------------- //
//重载错误参数输出
static inline void Print_Error_Parameter(int* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}
	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}
static inline void Print_Error_Parameter(float* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}
	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}
static inline void Print_Error_Parameter(std::string* info, const char* key, const char* val) {
	if (info == NULL) {
		std::cout << "[ " << key << " ] : - " << val << " - Read Error" << std::endl;
		Parameter_Error = TRUE;
	}

	if (Print_Parameter_Debug) {
		std::cout << "[ " << key << " ] \t- " << val << " = " << *info << std::endl;
	}
}

// ----------------------------------  重载加载参数  ---------------------------------- //
//加载节点参数
static inline char* Load_Str_Parameter(const char* Key, const char* Val) {
	char temp[MAX_PATH];
	GetPrivateProfileStringA(Key, Val, NULL, temp, MAX_PATH, cfg_info.Ini_Path.c_str());
	return temp;
}

//加载Int参数
static inline void Load_Node_Parameter(int* info, const char* Key, const char* Val) {
	*info = GetPrivateProfileIntA(Key, Val, NULL, cfg_info.Ini_Path.c_str());
	Print_Error_Parameter(info, Key, Val);
}

//加载Float参数
static inline void Load_Node_Parameter(float* info, const char* Key, const char* Val) {
	*info = strtod(Load_Str_Parameter(Key, Val), NULL);
	Print_Error_Parameter(info, Key, Val);

}

//加载String参数
static inline void Load_Node_Parameter(std::string* info, const char* Key, const char* Val) {

	*info = Load_Str_Parameter(Key, Val);
	//std::cout << *info << std::endl;
	Print_Error_Parameter(info, Key, Val);
}

//特殊节点,需要转字符编码
static inline void Load_Node_Parameter_Name(std::string* info, const char* Key, const char* Val) {

	char temp[MAX_PATH];
	GetPrivateProfileStringA(Key, Val, NULL, temp, MAX_PATH, cfg_info.Ini_Path.c_str());
	*info = UTF8ToAnsi(temp);
	//std::cout << *info << std::endl;
	Print_Error_Parameter(info, Key, Val);
}


// ----------------------------------  节点参数分装  ---------------------------------- //
static void Load_Check_Parameter(int idx) {
	//检查
	Load_Node_Parameter(&cfg_info.Check.checking_Version, cfg_info.Node[idx], cfg_info.Check_val[0]);
}

static void Load_Move_Parameter(int idx) {
	//移动
	Load_Node_Parameter(&cfg_info.Move.move_manner, cfg_info.Node[idx], cfg_info.Move_val[0]);
	Load_Node_Parameter(&cfg_info.Move.comx, cfg_info.Node[idx], cfg_info.Move_val[1]);
}


static void Load_Fov_Parameter(int idx) {
	//FOV 
	Load_Node_Parameter(&cfg_info.Fov.fov_off, cfg_info.Node[idx], cfg_info.Fov_val[0]);
	Load_Node_Parameter(&cfg_info.Fov.game_HFOV, cfg_info.Node[idx], cfg_info.Fov_val[1]);
	Load_Node_Parameter(&cfg_info.Fov.game_VFOV, cfg_info.Node[idx], cfg_info.Fov_val[2]);
	Load_Node_Parameter(&cfg_info.Fov.game_x_pixel, cfg_info.Node[idx], cfg_info.Fov_val[3]);
	Load_Node_Parameter(&cfg_info.Fov.game_y_pixel, cfg_info.Node[idx], cfg_info.Fov_val[4]);
}

static void Load_Pid_Parameter(int idx) {
	//PID
	Load_Node_Parameter(&cfg_info.Pid.pid_off, cfg_info.Node[idx], cfg_info.Pid_val[0]);
	Load_Node_Parameter(&cfg_info.Pid.kp_x, cfg_info.Node[idx], cfg_info.Pid_val[1]);
	Load_Node_Parameter(&cfg_info.Pid.ki_x, cfg_info.Node[idx], cfg_info.Pid_val[2]);
	Load_Node_Parameter(&cfg_info.Pid.kd_x, cfg_info.Node[idx], cfg_info.Pid_val[3]);
	Load_Node_Parameter(&cfg_info.Pid.kp_y, cfg_info.Node[idx], cfg_info.Pid_val[4]);
	Load_Node_Parameter(&cfg_info.Pid.ki_y, cfg_info.Node[idx], cfg_info.Pid_val[5]);
	Load_Node_Parameter(&cfg_info.Pid.kd_y, cfg_info.Node[idx], cfg_info.Pid_val[6]);
	Load_Node_Parameter(&cfg_info.Pid.sample_num, cfg_info.Node[idx], cfg_info.Pid_val[7]);
}

static void Load_Aim_Parameter(int idx) {
	//Aim
	Load_Node_Parameter(&cfg_info.Aim.range_top, cfg_info.Node[idx], cfg_info.Aim_val[0]);
	Load_Node_Parameter(&cfg_info.Aim.range_bottom, cfg_info.Node[idx], cfg_info.Aim_val[1]);
	Load_Node_Parameter(&cfg_info.Aim.range_left, cfg_info.Node[idx], cfg_info.Aim_val[2]);
	Load_Node_Parameter(&cfg_info.Aim.range_right, cfg_info.Node[idx], cfg_info.Aim_val[3]);
	Load_Node_Parameter(&cfg_info.Aim.cla_off, cfg_info.Node[idx], cfg_info.Aim_val[4]);
	Load_Node_Parameter(&cfg_info.Aim.label_chose, cfg_info.Node[idx], cfg_info.Aim_val[5]);
	Load_Node_Parameter(&cfg_info.Aim.move_offset, cfg_info.Node[idx], cfg_info.Aim_val[6]);
}

static void Load_Fire_Parameter(int idx) {
	//自动开火
	Load_Node_Parameter(&cfg_info.Fire.auto_off, cfg_info.Node[idx], cfg_info.Fire_val[0]);
	Load_Node_Parameter(&cfg_info.Fire.auto_method, cfg_info.Node[idx], cfg_info.Fire_val[1]);
	Load_Node_Parameter(&cfg_info.Fire.auto_sleep, cfg_info.Node[idx], cfg_info.Fire_val[2]);
}

static void Load_Key_Parameter(int idx) {
	//按键
	Load_Node_Parameter(&cfg_info.Key.aim_off, cfg_info.Node[idx], cfg_info.Key_val[0]);
	Load_Node_Parameter(&cfg_info.Key.key_method, cfg_info.Node[idx], cfg_info.Key_val[1]);
	Load_Node_Parameter(&cfg_info.Key.button_key1, cfg_info.Node[idx], cfg_info.Key_val[2]);
	Load_Node_Parameter(&cfg_info.Key.button_key2, cfg_info.Node[idx], cfg_info.Key_val[3]);
	Load_Node_Parameter(&cfg_info.Key.end_key, cfg_info.Node[idx], cfg_info.Key_val[4]);
}

static void Load_Pred_Parameter(int idx) {
	//推理
	Load_Node_Parameter(&cfg_info.Pred.engine_path, cfg_info.Node[idx], cfg_info.Pred_val[0]);
	Load_Node_Parameter(&cfg_info.Pred.frame, cfg_info.Node[idx], cfg_info.Pred_val[1]);
	Load_Node_Parameter(&cfg_info.Pred.conf, cfg_info.Node[idx], cfg_info.Pred_val[2]);
	Load_Node_Parameter(&cfg_info.Pred.iou, cfg_info.Node[idx], cfg_info.Pred_val[3]);
	Load_Node_Parameter(&cfg_info.Pred.sleep, cfg_info.Node[idx], cfg_info.Pred_val[4]);
}

static void Load_Windows_Parameter(int idx) {
	//窗口
	Load_Node_Parameter(&cfg_info.Windows.capture_method, cfg_info.Node[idx], cfg_info.Win_val[0]);
	Load_Node_Parameter(&cfg_info.Windows.win32_method, cfg_info.Node[idx], cfg_info.Win_val[1]);
	//特殊节点，需要单独处理
	Load_Node_Parameter_Name(&cfg_info.Windows.win32_name, cfg_info.Node[idx], cfg_info.Win_val[2]);
	Load_Node_Parameter(&cfg_info.Windows.show, cfg_info.Node[idx], cfg_info.Win_val[3]);
}
static void Load_Other_Parameter(int idx) {
	//其他
	Load_Node_Parameter(&cfg_info.Other.console_refresh, cfg_info.Node[idx], cfg_info.Other_val[0]);
}


static bool Load_Parameter() {
	//分装加载节点参数
	Load_Check_Parameter(0);
	Load_Move_Parameter(1);
	Load_Fov_Parameter(2);
	Load_Pid_Parameter(3);
	Load_Aim_Parameter(4);
	Load_Fire_Parameter(5);
	Load_Key_Parameter(6);
	Load_Pred_Parameter(7);
	Load_Windows_Parameter(8);
	Load_Other_Parameter(9);

	return true;
}
// ----------------------------------  Init  ---------------------------------- //
//加载ini参数
static bool Init_Parameter() {
	//加载ini参数
	if (!Load_Parameter()) {
		return false;
	}
	////验证engine文件是否存在
	//if (!Is_File(cfg_info.Pred.engine_path)) {
	//	std::cout << "找不到[ " << cfg_info.Pred.engine_path << " ]文件,检查文件是否存在和ini路径参数是否正确(严格检查)" << std::endl;
	//	system("pause");
	//	return false;
	//}
	return true;
}


bool Init_config() {
	//检查是否存在ini文件
	if (!Check_ini) {
		std::cout << "Check_Head: config.ini 文件不存在" << std::endl;
		return false;
	}

	//设置ini文件路径
	if (!Set_Config_Path()) {
		std::cout << "Set_Config_Path: 设置config.ini路径失败" << std::endl;
		return false;
	}
	//加载ini参数
	if (!Init_Parameter()) {
		std::cout << "Init_Parameter: 加载ini参数加载失败" << std::endl;
		return false;
	}

	//是否存在读取错误参数
	if (Parameter_Error) {
		std::cout << "Init_Parameter: 有ini参数读取错误,检查参数" << std::endl;
		return false;
	}
	std::cout << "Load config.ini Parameter PASS..." << std::endl;
	return true;
}

// ----------------------------------  Updata Thread  ---------------------------------- //

long old_modify_time = 0;

static inline bool Get_Last_File_Time() {
	//获取最后修改时间
	int fd = 0;
	int size_fp = 0;
	struct stat buff {};

	FILE* fp = fopen(cfg_info.Ini_Path.c_str(), "r");
	if (fp == NULL) {
		std::cout << "打开文件失败" << std::endl;
		return false;
	}

	//获取文件描述符
	fd = fileno(fp);
	//获取文件状态
	fstat(fd, &buff);
	//获取文件大小
	size_fp = buff.st_size;
	//获取文件修改时间
	long modify_time = buff.st_mtime;
	fclose(fp);

	if (old_modify_time != modify_time){
		old_modify_time = modify_time;
		return true;
	}
	return false;
}

//动态读取参数
static inline bool Load_Updata_Parameter() {

	//固定节点读取
	Load_Fov_Parameter(2);
	Load_Pid_Parameter(3);
	Load_Aim_Parameter(4);
	Load_Fire_Parameter(5);
	Load_Key_Parameter(6);
	Load_Other_Parameter(9);

	//分散节点读取
	//Pred
	Load_Node_Parameter(&cfg_info.Pred.conf, cfg_info.Node[7], cfg_info.Pred_val[2]);
	Load_Node_Parameter(&cfg_info.Pred.iou, cfg_info.Node[7], cfg_info.Pred_val[3]);
	Load_Node_Parameter(&cfg_info.Pred.sleep, cfg_info.Node[7], cfg_info.Pred_val[4]);
	//win
	Load_Node_Parameter(&cfg_info.Windows.show, cfg_info.Node[8], cfg_info.Win_val[3]);

	return true;
}


//动态读取线程
bool dynamic_thread_loop() {

	while (!Thread_Should_Stop()) {
		//获取最后修改时间
		if (Get_Last_File_Time()){
			//加载参数
			Load_Updata_Parameter();
		}
		//延时检查
		Sleep(1800);
	}
	std::cout << "动态读取线程退出" << std::endl;
	return true;
}
