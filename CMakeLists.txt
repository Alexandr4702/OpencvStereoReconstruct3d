cmake_minimum_required(VERSION 3.5)

project(calibrate3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED )
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories( ${OPENGL_INCLUDE_DIRS})

add_executable(calibrate3D stereo_calibrate.cpp)
target_link_libraries( calibrate3D ${OpenCV_LIBS} -lpthread -lstdc++fs)

add_executable(reconstruct stereoReconstruct.cpp)
target_link_libraries( reconstruct ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} -lGL -lGLU -lsfml-graphics -lsfml-window -lsfml-system)

# add_executable(captureImageRpi captureImageRpi.cpp)
# target_link_libraries( captureImageRpi ${OpenCV_LIBS} -larducam_mipicamera -lpthread -lstdc++fs)
