cmake_minimum_required(VERSION 3.18)		# cmake 最低版本，3.18具有新特性

project(TensorRT_SF) 	# 项目名称

# -- 改成自己的路径
set(OpenCV_DIR "D:\\opencv\\build")   # 你自己的opecv根路径

# -- 设置标准
add_definitions(-DAPI_EXPORTS)			# 这行解决警告
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)		# 设置可分离编译，在指定生成cu文件之前
set(CMAKE_CXX_STANDARD 11)			# 设置编译器版本
set(CMAKE_BUILD_TYPE  Release)			# 设置debug or Release


# -- 设置多线程
set(THREADS_PREFER_PTHREAD_FLAG ON)		# 设置线程首选标志
find_package(Threads)				# 引入线程库
	
# -- CUDA设置
set(CMAKE_CUDA_ARCHITECTURES  75)		# 修改为自己的显卡算力,可以写多个显卡,30系显卡必须要11.1以上的cuda
find_package(CUDA REQUIRED)			# 引入CUDA依赖包
include_directories(${CUDA_INCLUDE_DIRS})		# 指定查找 CUDA 的include路径
enable_language(CUDA)  				# 激活CUDA语言支持，则无需在 vs 中设置 cuda 路径


# -- 项目设置
include_directories(${PROJECT_SOURCE_DIR}/include)	# 指定项目的include目录
LINK_DIRECTORIES(${PROJECT_SOURCE_DIR}/lib)		# 指定项目的lib路径

# --  用于解决错误：标识符“__builtin_ia32_mwaitx”未定义 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED") 

# -- 设置 opencv
find_package(OpenCV QUIET)				# 引入opecv库
include_directories(${OpenCV_INCLUDE_DIRS})		#  opencv  的 include目录

# 设置头文件
file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp)
source_group("Include" FILES ${CURRENT_HEADERS}) 

# 设置cpp文件
file(GLOB_RECURSE CPP_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# -- 指定生成的文件 
add_executable(Yolo_V5 ${CPP_FILES} ${CURRENT_HEADERS})

# -- 链接依赖
target_link_libraries(Yolo_V5 "nvinfer.lib" "nvinfer_plugin.lib")  	# 链接nvinfer.lib  nvinfer_plugin.lib等依赖
target_link_libraries(Yolo_V5 ${CUDA_LIBRARIES})  		# 链接CUDA依赖
target_link_libraries(Yolo_V5 ${OpenCV_LIBS})     		# 链接opencv依赖
target_link_libraries(Yolo_V5 "onnxruntime.lib")  		# 链接ort的lib依赖
target_link_libraries(Yolo_V5 Threads::Threads)     		# 链接线程依赖
target_link_libraries(Yolo_V5 "msdk.lib")     			# 链接易键鼠
target_link_libraries(Yolo_V5 "ntdll.lib")     			# 
