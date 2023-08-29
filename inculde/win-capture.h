#pragma once
#include<windows.h>
#include <opencv2/opencv.hpp>

bool Init_BitBlt();
bool Free_BitBlt();

struct win_capture {

	int size;
	cv::Mat win_img;
	HWND window_hwnd;
	HDC Window_DC;
	HDC compatible_DC;
	HBITMAP BitMap;
};

