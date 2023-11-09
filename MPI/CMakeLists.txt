cmake_minimum_required(VERSION 3.12)

project(MultipleExecutables LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

# Add an executable for mergesort.cpp
add_executable(mergesort mergesort.cpp)
target_include_directories(mergesort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)  # Adjust as per your project structure

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(mergesort PRIVATE MPI::MPI_CXX)
target_link_libraries(mergesort PRIVATE caliper)

# Add an executable for enumeration.cpp
add_executable(enumeration enumeration.cpp)
target_include_directories(enumeration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)  # Adjust as per your project structure

target_link_libraries(enumeration PRIVATE MPI::MPI_CXX)
target_link_libraries(enumeration PRIVATE caliper)

# Add an executable for bitonic.cpp
add_executable(bitonic bitonic.cpp)
target_include_directories(bitonic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)  # Adjust as per your project structure

target_link_libraries(bitonic PRIVATE MPI::MPI_CXX)
target_link_libraries(bitonic PRIVATE caliper)

# # Add an executable for oddeven.cpp
# add_executable(oddeven oddeven.cpp)
# target_include_directories(oddeven PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)  # Adjust as per your project structure

# target_link_libraries(oddeven PRIVATE MPI::MPI_CXX)
# target_link_libraries(oddeven PRIVATE caliper)
