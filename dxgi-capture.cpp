
#include "sf-info.h"
#include"dxgi-capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")


struct dxgi_info dxgi_data;

static bool Desc_Should_Init = false;       //Desc是否需要初始化


static inline bool Init_Device() {
	D3D_FEATURE_LEVEL FeatureLevel;

	//创建d3d11设备
	HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE,
		nullptr, 0, NULL, NULL, D3D11_SDK_VERSION,
		&dxgi_data.device, &FeatureLevel, &dxgi_data.Context);
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool GetDevice2() {
	//根据设备创建设备2
	HRESULT hr = dxgi_data.device->QueryInterface(__uuidof(IDXGIDevice2), (void**)&dxgi_data.DxgiDevice2);
	if (FAILED(hr)) {
		return false;
	}
	//设置最大帧数延迟
	dxgi_data.DxgiDevice2->SetMaximumFrameLatency(1);
	return true;
}

static inline bool Get_DXGI_Adapter() {
	//获取设备适配器
	HRESULT hr = dxgi_data.DxgiDevice2->GetParent(__uuidof(IDXGIAdapter), reinterpret_cast<void**>(&dxgi_data.DxgiAdapter));
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_DxgiOutput() {
	//枚举显示器
	HRESULT hr = dxgi_data.DxgiAdapter->EnumOutputs(dxgi_data.output, &dxgi_data.DxgiOutput);
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_Output1_Interface() {
	//获取显示器接口
	HRESULT hr = dxgi_data.DxgiOutput->QueryInterface(__uuidof(dxgi_data.DxgiOutput1), reinterpret_cast<void**>(&dxgi_data.DxgiOutput1));	//查询接口
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Get_desk_Duplicate() {
	//获取桌面副本的Desk信息
	HRESULT hr = dxgi_data.DxgiOutput1->DuplicateOutput(dxgi_data.device, &dxgi_data.desktopDupl);
	//部分win10出现此错误，未知原因
	if (hr == DXGI_ERROR_UNSUPPORTED){
		std::cout << "Get_desk_Duplicate: 设备或驱动程序不支持请求的功能" << std::endl;
		return false;
	}
	if (FAILED(hr)) {
		return false;
	}
	return true;
}

static inline bool Init_Interface() {
	if (!GetDevice2()) {
		std::cout << "Init_Interface: 获取Device2失败" << std::endl;
		return false;
	}
	if (!Get_DXGI_Adapter()) {
		std::cout << "Init_Interface: 获取显卡适配器失败" << std::endl;
		return false;
	}
	if (!Get_DxgiOutput()) {
		std::cout << "Init_Interface: 获取显示器索引失败" << std::endl;
		return false;
	}
	if (!Get_Output1_Interface()) {
		std::cout << "Init_Interface: 获取Output1接口失败" << std::endl;
		return false;
	}
	if (!Get_desk_Duplicate()) {
		std::cout << "Init_Interface: 获取桌面副本接口失败" << std::endl;
		return false;
	}
	return true;
}

static inline bool Get_Next_Frame() {

	//获取后缓冲
	HRESULT hr = dxgi_data.desktopDupl->AcquireNextFrame(0, &dxgi_data.frameInfo, &dxgi_data.desktopResource);
	//桌面无变化时，dxgi休眠，返回无任何参数的true即可
	if (hr == DXGI_ERROR_WAIT_TIMEOUT) {		
		//std::cout << "窗口无变化" << std::endl;
		return true;
	}

	//如果AcquireNextFrame 副本丢失,需要重新初始化 desktopDupl，常见于cf启动游戏
	if (FAILED(hr)) {	
		std::cout << "Get_Next_Frame: desktopDupl丢失，重新初始化" << std::endl;
		if (!Get_desk_Duplicate()) {
			std::cout << "Get_Next_Frame: desktopDupl重新初始化失败" << std::endl;
			return false;
		}else {
			std::cout << "Get_Next_Frame: desktopDupl重初初始化成功" << std::endl;
			Desc_Should_Init = true;	//更新Desk信息
			return true;
		}
	}
	return true;
}

static inline bool Get_Frame_prt() {
	//获取后缓冲指针 
	HRESULT hr = dxgi_data.desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&dxgi_data.acquiredDesktopImage));
	if (FAILED(hr)) {
		return false;
	}

	return true;
}

static bool Init_Desc() {
	//初始化桌面副本信息
	dxgi_data.acquiredDesktopImage->GetDesc(&dxgi_data.dataDesc);
	dxgi_data.dataDesc.Usage = D3D11_USAGE_STAGING;
	dxgi_data.dataDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
	dxgi_data.dataDesc.BindFlags = 0;
	dxgi_data.dataDesc.MiscFlags = 0;
	dxgi_data.dataDesc.MipLevels = 1;
	dxgi_data.dataDesc.SampleDesc.Count = 1;
	return true;
}

static inline bool Create_Tex() {
	//创建兼容的2d纹理
	dxgi_data.device->CreateTexture2D(&dxgi_data.dataDesc, NULL, &dxgi_data.copyDesktop);
	if (dxgi_data.copyDesktop == NULL) {
		return false;
	}
	return true;
}

static inline bool Unmap_and_Release() {
	dxgi_data.Context->Unmap(dxgi_data.copyDesktop, NULL);	//取消映射
	dxgi_data.copyDesktop->Release();		//释放纹理
	dxgi_data.desktopDupl->ReleaseFrame();	//释放后缓冲
	return true;
}

static inline bool DXG_Capture_Map() {

	//获取后缓冲
	if (!Get_Next_Frame()) {	//0ms
		return false;
	}
	//获取后缓冲的指针
	if (!Get_Frame_prt()) {	//0ms
		return false;
	}
	//只初始化1次
	if (Desc_Should_Init) {	//0ms
		if (Init_Desc()) {
			Desc_Should_Init = false;
		}
	}
	//创建2d纹理
	if (!Create_Tex()) {	//0ms
		return false;
	}

	//copy 0ms
	dxgi_data.Context->CopyResource(dxgi_data.copyDesktop, dxgi_data.acquiredDesktopImage);

	//将纹理映射到cpu 1ms
	dxgi_data.Context->Map(dxgi_data.copyDesktop, 0, D3D11_MAP_READ, 0, &dxgi_data.mappedResource);

	//转为Mat 0ms
	global_data.img = cv::Mat(int(dxgi_data.dataDesc.Height), int(dxgi_data.dataDesc.Width), CV_8UC4, dxgi_data.mappedResource.pData)(dxgi_data.rect);

	//4->3	1-3ms
	cv::cvtColor(global_data.img, global_data.img, cv::COLOR_BGRA2BGR);

	if (!Unmap_and_Release()) {	//0ms
		std::cout << "Unmap_and_Release: 释放错误" << std::endl;
		return false;
	}

	return true;
}

static inline bool DXGI_Screenshot_size() {

	//获取后缓冲
	if (Get_Next_Frame()) {	//0ms
		dxgi_data.desktopDupl->ReleaseFrame();	//释放后缓冲
	}
	//后缓冲的指针
	if (Get_Frame_prt()) {	//0ms
		dxgi_data.acquiredDesktopImage->Release();
	}

	//初始化Desk
	if (!Init_Desc()) {
		std::cout << "Create_Desc错误" << std::endl;
	}

	//屏幕大小
	global_data.window_width = int(dxgi_data.dataDesc.Width);
	global_data.window_height = int(dxgi_data.dataDesc.Height);

	//屏幕中心点
	global_data.cx = int(dxgi_data.dataDesc.Width * 0.5f);
	global_data.cy = int(dxgi_data.dataDesc.Height * 0.5f);

	//截图原点
	global_data.origin_x = global_data.cx - int(global_data.Input_Dim[2] * 0.5f);
	global_data.origin_y = global_data.cy - int(global_data.Input_Dim[3] * 0.5f);

	//创建截图范围
	dxgi_data.rect = cv::Rect(global_data.origin_x, global_data.origin_y, global_data.Input_Dim[2], global_data.Input_Dim[3]);

	return true;
}

bool Free_dxgi() {

	if (dxgi_data.device) {
		dxgi_data.device->Release();	//释放
		dxgi_data.device = nullptr;	//防野
	}
	if (dxgi_data.Context) {
		dxgi_data.Context->Release();
		dxgi_data.Context = nullptr;
	}

	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//释放
		dxgi_data.DxgiDevice2 = nullptr;	//防野
	}
	if (dxgi_data.DxgiDevice2) {
		dxgi_data.DxgiDevice2->Release();	//释放
		dxgi_data.DxgiDevice2 = nullptr;	//防野
	}
	if (dxgi_data.DxgiAdapter) {
		dxgi_data.DxgiAdapter->Release();
		dxgi_data.DxgiAdapter = nullptr;
	}
	if (dxgi_data.DxgiOutput) {
		dxgi_data.DxgiOutput->Release();
		dxgi_data.DxgiOutput = nullptr;
	}
	if (dxgi_data.DxgiOutput1) {
		dxgi_data.DxgiOutput1->Release();
		dxgi_data.DxgiOutput1 = nullptr;
	}
	if (dxgi_data.desktopDupl) {
		dxgi_data.desktopDupl->Release();
		dxgi_data.desktopDupl = nullptr;
	}
	if (dxgi_data.desktopResource) {
		dxgi_data.desktopResource->Release();
		dxgi_data.desktopResource = nullptr;
	}

	if (dxgi_data.acquiredDesktopImage) {
		dxgi_data.acquiredDesktopImage->Release();
		dxgi_data.acquiredDesktopImage = nullptr;
	}

	//越界错误，阻塞
	//if (dxgi_data.copyDesktop) {
	//	dxgi_data.copyDesktop->Release();
	//	dxgi_data.copyDesktop = nullptr;
	//}

	return true;
}

bool Init_dxgi() {
	//创建d3d11设备
	if (!Init_Device()) {
		std::cout << "Init_dxgi: 创建D3D11设备失败" << std::endl;
		return false;
	}

	//创建d3d接口
	if (!Init_Interface()) {
		std::cout << "Init_dxgi: 初始化接口失败" << std::endl;
		return false;
	}

	//设置截图大小
	if (!DXGI_Screenshot_size()) {
		std::cout << "Init_dxgi: 设置截图范围失败" << std::endl;
		return false;
	}

	//标志需要初始化Desc
	Desc_Should_Init = true;

	//函数地址
	//global_data.capture_map存储的是DXG_Capture_Map函数的内存地址
	//global_data.capture_map() 等效于 DXG_Capture_Map()
	global_data.capture_map = DXG_Capture_Map;
	global_data.capture_free = Free_dxgi;

	return true;
}

