cmake_minimum_required(VERSION 3.18)

# PROJECT CONFIGURATION
project(DSfM LANGUAGES CXX CUDA VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})

message(STATUS "Using CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")

# Set build type to 'RelWithDebInfo' if one was not specified by the user
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS RelWithDebInfo Release Debug MinSizeRel)
  message(STATUS "Setting build type to ${CMAKE_BUILD_TYPE}, as none was specified\n")
else()
  message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")
endif()

# Add --extended-lambda to CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place the library built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" CACHE STRING "The CMake module path used for this project")

# Packages
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

find_package(NCCL REQUIRED)
include_directories(${NCCL_INCLUDE_DIRS})

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

include_directories(.)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Subdirectories
set(DSfM_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sfm/ CACHE PATH "Path to top-level directory of DSfM (the one containing CMakeLists.txt)")
set(EXAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/ CACHE PATH "Path to top-level directory of DSfM (the one containing CMakeLists.txt)")


add_subdirectory(${DSfM_DIR})
add_subdirectory(${EXAMPLE_DIR})