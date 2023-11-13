# CleanAll.cmake
# Clean all build artifacts except specified file types

file(GLOB_RECURSE ALL_FILES ${CMAKE_BINARY_DIR}/*)
foreach(FILE ${ALL_FILES})
    get_filename_component(EXTENSION ${FILE} EXT)
    if(NOT EXTENSION STREQUAL ".cu" AND NOT EXTENSION STREQUAL ".cpp" AND NOT EXTENSION STREQUAL ".txt" AND NOT EXTENSION STREQUAL ".cali" AND NOT EXTENSION STREQUAL ".sh" AND NOT EXTENSION STREQUAL ".grace_job" AND NOT EXTENSION STREQUAL ".cmake")
        file(REMOVE ${FILE})
    endif()
endforeach()

file(GLOB_RECURSE ALL_DIRS ${CMAKE_BINARY_DIR}/*/)
foreach(DIR ${ALL_DIRS})
    file(REMOVE_RECURSE ${DIR})
endforeach()
