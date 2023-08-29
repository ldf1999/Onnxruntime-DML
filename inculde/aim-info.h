#pragma once

#include<windows.h>

DWORD WINAPI aim_func();	//主函数调用

struct PID {

	float SetSpeed = 0;			//记录本次移动距离
	float Error = 0;			//与上一次移动的误差
	float Err_Next = 0;			//
	float ActualSpeed = 0;		//上一次的移动增量距离
	float IncrementSpeed = 0;	//计算后的移动距离
	float Err_last = 0;			//计算后的移动距离
	int Sample_frame = 0;		//采样帧

};
