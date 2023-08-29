#pragma once
#include <winternl.h>

bool Init_Move();


// -----------------------------------   SendInput移动  -----------------------------------
struct Send_info {
	INPUT input;
};


// -----------------------------------   飞易来/易键鼠移动   -----------------------------------
struct FeiOrYi_info {
	HANDLE FOY_hwnd = NULL;
};

// -----------------------------------   kmbox移动  -----------------------------------
struct KmBox_info{
	HANDLE com_hwnd = NULL;
};


// -----------------------------------   罗技移动  -----------------------------------
typedef struct {
	char button;
	char x;
	char y;
	char wheel;
	char unk1;
} MOUSE_IO;

struct Logitech_info {
	MOUSE_IO io{};
	HANDLE LG_input = NULL;
	IO_STATUS_BLOCK LG_io{};
	NTSTATUS LG_status{};
	//BOOL LG_found_mouse;
};
