#pragma once
#include<opencv2/opencv.hpp>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>
#include <dxgi1_5.h>


bool Init_dxgi();
bool Free_dxgi();

struct dxgi_info{
	cv::Rect rect;
	UINT output = 0;	//显示器索引
	ID3D11Device* device = nullptr;	//ID3D11 设备
	ID3D11DeviceContext* Context = nullptr;	//ID3D11 设备上下文
	IDXGIDevice2* DxgiDevice2 = nullptr;
	IDXGIAdapter* DxgiAdapter = nullptr;
	IDXGIOutput* DxgiOutput = nullptr;
	IDXGIOutput1* DxgiOutput1 = nullptr;

	IDXGIOutputDuplication* desktopDupl = nullptr;
	DXGI_OUTDUPL_FRAME_INFO frameInfo;	//DXGI 输出帧信息
	IDXGIResource* desktopResource = nullptr;	//桌面纹理
	ID3D11Texture2D* acquiredDesktopImage = nullptr;	//获取的桌面纹理
	D3D11_TEXTURE2D_DESC dataDesc = {NULL};
	ID3D11Texture2D* copyDesktop = nullptr;	//拷贝到的纹理
	D3D11_MAPPED_SUBRESOURCE mappedResource;	//D3D11 GPU映射都cpu
};
