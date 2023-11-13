cmake_minimum_required(VERSION 3.12)

project(MultipleExecutables LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)
message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})
# Add an executable for bitonic
add_executable(bitonic bitonic.cu)
target_include_directories(bitonic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(bitonic PRIVATE MPI::MPI_CXX)
target_link_libraries(bitonic PRIVATE caliper)
# target_link_libraries(bitonic PRIVATE adiak)

# Add an executable for merge
add_executable(merge merge.cu)
target_include_directories(merge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(merge PRIVATE MPI::MPI_CXX)
target_link_libraries(merge PRIVATE caliper)
# target_link_libraries(merge PRIVATE adiak)

# Add an executable for oddeven
add_executable(oddeven oddeven.cu)
target_include_directories(oddeven PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(oddeven PRIVATE MPI::MPI_CXX)
target_link_libraries(oddeven PRIVATE caliper)
# target_link_libraries(oddeven PRIVATE adiak)

# Add an executable for enumeration
add_executable(enumeration enumeration.cu)
target_include_directories(enumeration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_link_libraries(enumeration PRIVATE MPI::MPI_CXX)
target_link_libraries(enumeration PRIVATE caliper)
# target_link_libraries(enumeration PRIVATE adiak)
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/CleanAll.cmake
    COMMENT "Cleaning build artifacts"
)