# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/aurkorouth/research/spack/opt/spack/linux-ubuntu20.04-icelake/gcc-9.4.0/cmake-3.27.4-fsub5zsbtum27p7qbbe54lqssydw6m4k/bin/cmake

# The command to remove a file.
RM = /home/aurkorouth/research/spack/opt/spack/linux-ubuntu20.04-icelake/gcc-9.4.0/cmake-3.27.4-fsub5zsbtum27p7qbbe54lqssydw6m4k/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aurkorouth/435-Project/MPI

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aurkorouth/435-Project/MPI

# Include any dependencies generated for this target.
include CMakeFiles/mergesort.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mergesort.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mergesort.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mergesort.dir/flags.make

CMakeFiles/mergesort.dir/mergesort.cpp.o: CMakeFiles/mergesort.dir/flags.make
CMakeFiles/mergesort.dir/mergesort.cpp.o: mergesort.cpp
CMakeFiles/mergesort.dir/mergesort.cpp.o: CMakeFiles/mergesort.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/aurkorouth/435-Project/MPI/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mergesort.dir/mergesort.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mergesort.dir/mergesort.cpp.o -MF CMakeFiles/mergesort.dir/mergesort.cpp.o.d -o CMakeFiles/mergesort.dir/mergesort.cpp.o -c /home/aurkorouth/435-Project/MPI/mergesort.cpp

CMakeFiles/mergesort.dir/mergesort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mergesort.dir/mergesort.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aurkorouth/435-Project/MPI/mergesort.cpp > CMakeFiles/mergesort.dir/mergesort.cpp.i

CMakeFiles/mergesort.dir/mergesort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mergesort.dir/mergesort.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aurkorouth/435-Project/MPI/mergesort.cpp -o CMakeFiles/mergesort.dir/mergesort.cpp.s

# Object files for target mergesort
mergesort_OBJECTS = \
"CMakeFiles/mergesort.dir/mergesort.cpp.o"

# External object files for target mergesort
mergesort_EXTERNAL_OBJECTS =

mergesort: CMakeFiles/mergesort.dir/mergesort.cpp.o
mergesort: CMakeFiles/mergesort.dir/build.make
mergesort: /usr/local/lib/libcaliper.so.2.11.0-dev
mergesort: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
mergesort: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
mergesort: CMakeFiles/mergesort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/aurkorouth/435-Project/MPI/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mergesort"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mergesort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mergesort.dir/build: mergesort
.PHONY : CMakeFiles/mergesort.dir/build

CMakeFiles/mergesort.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mergesort.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mergesort.dir/clean

CMakeFiles/mergesort.dir/depend:
	cd /home/aurkorouth/435-Project/MPI && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aurkorouth/435-Project/MPI /home/aurkorouth/435-Project/MPI /home/aurkorouth/435-Project/MPI /home/aurkorouth/435-Project/MPI /home/aurkorouth/435-Project/MPI/CMakeFiles/mergesort.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/mergesort.dir/depend

