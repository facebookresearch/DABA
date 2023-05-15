# Enable faster instruction sets (SIMD/AVX)
set(ENABLE_FAST_INSTRUCTIONS ON CACHE BOOL "Enable faster instruction sets (SIMD/AVX)")

# Directories
set(DSfM_INCLUDE_DIR ${PROJECT_SOURCE_DIR})
set(DSfM_INCLUDES ${DSfM_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} CACHE INTERNAL "")

set(DSfM_SRC_DIR ${PROJECT_SOURCE_DIR})
file(GLOB_RECURSE DSfM_HDRS ${DSfM_SRC_DIR} "*.h" "*.cuh")
file(GLOB_RECURSE DSfM_SRCS ${DSfM_SRC_DIR} "*.cpp" "*.cu")

add_library(DSfM STATIC ${DSfM_HDRS} ${DSfM_SRCS})
target_include_directories(DSfM PUBLIC ${DSfM_INCLUDE_DIR})
target_link_libraries(DSfM PUBLIC ${CUDA_LIBRARIES} ${NCCL_LIBRARIES} ${MPI_CXX_LIBRARIES} glog cuda)

# Add add entry for this project into CMake's package registry, so that this project can be found by other CMake projects
export(PACKAGE DSfM)
# Create a configuration file for this project, so that it can be imported by other CMake projects
export(TARGETS DSfM FILE DSfMConfig.cmake)