cmake_minimum_required(VERSION 3.9.0)

project(testNcnn VERSION 1.0)

# >>> build type 
# set(CMAKE_BUILD_TYPE "Debug")				# 指定生成的版本
# set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
# set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
# <<<


# >>> CXX11 
set(CMAKE_CXX_STANDARD 11)				# C++ 11 编译器
SET(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# <<< CXX11

# Ncnn
if(WIN32)
    set(NCNN_ROOT "D:/ncnn/ncnn_install/Release/ncnn_install_no_vulkan")
    set(NCNN_ROOT_DEBUG "D:/ncnn/ncnn_install/Debug/ncnn_install_no_vulkan")
	
    set(NCNN_LIBRARIES 
        "${NCNN_ROOT}/lib/ncnn.lib"
    CACHE INTERNAL "")
    set(NCNN_LIBRARIES_DEBUG
        "${NCNN_ROOT_DEBUG}/lib/ncnnd.lib"
    CACHE INTERNAL "")
	
	set(NCNN_INCLUDE_PATH "${NCNN_ROOT_DEBUG}/include" CACHE INTERNAL "")
    
elseif(UNIX)
    
endif()

# OpenCV
if(WIN32)
    set(OPENCV_ROOT "D:/opencv-4.5.4/Release")
    set(OPENCV_ROOT_DEBUG "D:/opencv-4.5.4/Debug")
	
    set(OPENCV_LIBRARIES 
        "${OPENCV_ROOT}/lib/x64/opencv_world454.lib"
    CACHE INTERNAL "")
    set(OPENCV_LIBRARIES_DEBUG
        "${OPENCV_ROOT_DEBUG}/lib/x64/opencv_world454d.lib"
    CACHE INTERNAL "")
	
    set(OPENCV_RUNTIME_LIBRARIES
        "${OPENCV_ROOT}/bin/opencv_world454.dll"
    CACHE INTERNAL "")
    set(OPENCV_RUNTIME_LIBRARIES_DEBUG
        "${OPENCV_ROOT_DEBUG}/bin/opencv_world454d.dll"
    CACHE INTERNAL "")
	set(OPENCV_INCLUDE_PATH "${OPENCV_ROOT}/include" CACHE INTERNAL "")
elseif(UNIX)
    set(OPENCV_ROOT "/home/opencv-4.5.4/Release")
    set(OPENCV_LIBRARIES 
        "${OPENCV_ROOT}/lib/Linux64/libopencv_core.so"
        "${OPENCV_ROOT}/lib/Linux64/libopencv_videoio.so"
        "${OPENCV_ROOT}/lib/Linux64/libopencv_highgui.so"
        "${OPENCV_ROOT}/lib/Linux64/libopencv_imgcodecs.so"
        "${OPENCV_ROOT}/lib/Linux64/libopencv_imgproc.so"
	    "${OPENCV_ROOT}/lib/Linux64/libopencv_dnn.so"
        "${OPENCV_ROOT}/lib/Linux64/libopencv_flann.so"
    CACHE INTERNAL "")
    set(OPENCV_LIBRARIES_DEBUG ${OPENCV_LIBRARIES} CACHE INTERNAL "")
    file(GLOB TMP_SO_FILES "${OPENCV_ROOT}/lib/Linux64/*.so*")
    set(OPENCV_RUNTIME_LIBRARIES ${TMP_SO_FILES} CACHE INTERNAL "")
    set(OPENCV_RUNTIME_LIBRARIES_DEBUG ${TMP_SO_FILES} CACHE INTERNAL "")
	set(OPENCV_INCLUDE_PATH "${OPENCV_ROOT}/include" CACHE INTERNAL "")
endif()

add_executable(${PROJECT_NAME} seg.h seg.cpp main.cpp testNcnn.cpp)
target_include_directories(${PROJECT_NAME}  
PRIVATE 
	${NCNN_INCLUDE_PATH}
    ${OPENCV_INCLUDE_PATH}
)
target_link_libraries(${PROJECT_NAME} 
PUBLIC
	debug ${NCNN_LIBRARIES_DEBUG}
	optimized ${NCNN_LIBRARIES}
    debug ${OPENCV_LIBRARIES_DEBUG}
	optimized ${OPENCV_LIBRARIES}
)   


if(WIN32)

set(DLL_FILES
    ${OPENCV_RUNTIME_LIBRARIES}
)

set(DLL_FILES_DEBUG 
    ${OPENCV_RUNTIME_LIBRARIES_DEBUG}
)

add_custom_command(
TARGET ${PROJECT_NAME}
COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<IF:$<CONFIG:Debug>,${DLL_FILES_DEBUG},${DLL_FILES}>" $<TARGET_FILE_DIR:${PROJECT_NAME}>
COMMAND_EXPAND_LISTS
)
endif()



    


