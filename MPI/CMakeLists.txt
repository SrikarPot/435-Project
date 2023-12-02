cmake_minimum_required(VERSION 3.12)

project(MultipleExecutables LANGUAGES CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

# Function to add an executable
function(add_custom_executable EXECUTABLE_NAME SOURCE_FILE)
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
    target_include_directories(${EXECUTABLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)  # Adjust as per your project structure
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE MPI::MPI_CXX)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE caliper)
    # Uncomment the next line if using Adiak
    # target_link_libraries(${EXECUTABLE_NAME} PRIVATE adiak)
endfunction()

# Add an executable for mergesort.cpp
add_custom_executable(merge merge.cpp)

# Add an executable for enumeration.cpp
add_custom_executable(enumeration enumeration.cpp)

# Add an executable for bitonic.cpp
add_custom_executable(bitonic bitonic.cpp)

# Add an executable for oddeven.cpp
add_custom_executable(oddeven oddeven.cpp)
