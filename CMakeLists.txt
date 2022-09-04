cmake_minimum_required(VERSION 3.5)

project(calibrate3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(calibrate3D stereo_calibrate.cpp)
add_executable(reconstruct stereoReconstruct.cpp)

target_link_libraries( calibrate3D ${OpenCV_LIBS} )
target_link_libraries( reconstruct ${OpenCV_LIBS} )
