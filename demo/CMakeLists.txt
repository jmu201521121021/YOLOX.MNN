cmake_minimum_required(VERSION 2.6)
project(yolox)
find_package(OpenCV)

add_definitions(-std=c++11)
add_definitions(-O3 -pthread)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/
                    ${PROJECT_SOURCE_DIR}/../3rd_party/libmnn/ubuntu/include)

link_directories( ${PROJECT_SOURCE_DIR}/../3rd_party/libmnn/ubuntu/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolox
        ${PROJECT_SOURCE_DIR}/main.cpp
        ${PROJECT_SOURCE_DIR}/YOLOXMNN.cpp
        )
target_link_libraries(yolox
                    ${OpenCV_LIBS}
                     MNN)


