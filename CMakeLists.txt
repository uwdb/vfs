cmake_minimum_required(VERSION 3.12)
project(vfs)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

find_package(FUSE REQUIRED)
find_package(CUDA REQUIRED)
find_package (Glog REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

include_directories(/home/bhaynes/lightdb/core)

# Set compiler flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FUSE_DEFINITIONS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFUSE_USE_VERSION=30")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -g3 -rdynamic -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -fvisibility=hidden -Wall  -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_COVERAGE "-DDEBUG --coverage -g3 -Wall -Wextra -Wno-unused-parameter")

file(GLOB_RECURSE SOURCE_FILES *.cc *.h)
add_library(lib${PROJECT_NAME} SHARED ${SOURCE_FILES})
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

message(INFO ${CUDA_LIBRARIES})
target_link_libraries(lib${PROJECT_NAME} ${FUSE_LIBRARIES} ${GLOG_LIBRARIES} stdc++fs)
target_link_libraries(${PROJECT_NAME} cudart ${CUDA_npp_LIBRARY} ${CUDA_nppi_LIBRARY} pthread /home/bhaynes/projects/CudaSift/build/libcudasift_static.a /home/bhaynes/projects/lightdb/cmake-build-relwithdebinfo/core/liblightdb.a libvfs)
