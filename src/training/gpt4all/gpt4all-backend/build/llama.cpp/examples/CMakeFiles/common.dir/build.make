# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build

# Include any dependencies generated for this target.
include llama.cpp/examples/CMakeFiles/common.dir/depend.make

# Include the progress variables for this target.
include llama.cpp/examples/CMakeFiles/common.dir/progress.make

# Include the compile flags for this target's objects.
include llama.cpp/examples/CMakeFiles/common.dir/flags.make

llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o: llama.cpp/examples/CMakeFiles/common.dir/flags.make
llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o: ../llama.cpp/examples/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/common.dir/common.cpp.o -c /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/common.cpp

llama.cpp/examples/CMakeFiles/common.dir/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/common.dir/common.cpp.i"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/common.cpp > CMakeFiles/common.dir/common.cpp.i

llama.cpp/examples/CMakeFiles/common.dir/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/common.dir/common.cpp.s"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/common.cpp -o CMakeFiles/common.dir/common.cpp.s

common: llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o
common: llama.cpp/examples/CMakeFiles/common.dir/build.make

.PHONY : common

# Rule to build all files generated by this target.
llama.cpp/examples/CMakeFiles/common.dir/build: common

.PHONY : llama.cpp/examples/CMakeFiles/common.dir/build

llama.cpp/examples/CMakeFiles/common.dir/clean:
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples && $(CMAKE_COMMAND) -P CMakeFiles/common.dir/cmake_clean.cmake
.PHONY : llama.cpp/examples/CMakeFiles/common.dir/clean

llama.cpp/examples/CMakeFiles/common.dir/depend:
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/CMakeFiles/common.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : llama.cpp/examples/CMakeFiles/common.dir/depend
