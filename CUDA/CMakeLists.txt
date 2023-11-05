cmake_minimum_required(VERSION 3.12)

project(merge LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

add_executable(merge merge.cu)

# Add the directory containing your header file to include directories
target_include_directories(merge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(merge PRIVATE MPI::MPI_CXX)
target_link_libraries(merge PRIVATE caliper)
