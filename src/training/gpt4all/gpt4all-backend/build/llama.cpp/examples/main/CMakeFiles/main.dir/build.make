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
include llama.cpp/examples/main/CMakeFiles/main.dir/depend.make

# Include the progress variables for this target.
include llama.cpp/examples/main/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include llama.cpp/examples/main/CMakeFiles/main.dir/flags.make

llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.o: llama.cpp/examples/main/CMakeFiles/main.dir/flags.make
llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.o: ../llama.cpp/examples/main/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.o"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/main.dir/main.cpp.o -c /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/main/main.cpp

llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/main/main.cpp > CMakeFiles/main.dir/main.cpp.i

llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/main/main.cpp -o CMakeFiles/main.dir/main.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS = \
"/home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o"

bin/main: llama.cpp/examples/main/CMakeFiles/main.dir/main.cpp.o
bin/main: llama.cpp/examples/CMakeFiles/common.dir/common.cpp.o
bin/main: llama.cpp/examples/main/CMakeFiles/main.dir/build.make
bin/main: llama.cpp/libllama.so
bin/main: llama.cpp/examples/main/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/main"
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
llama.cpp/examples/main/CMakeFiles/main.dir/build: bin/main

.PHONY : llama.cpp/examples/main/CMakeFiles/main.dir/build

llama.cpp/examples/main/CMakeFiles/main.dir/clean:
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : llama.cpp/examples/main/CMakeFiles/main.dir/clean

llama.cpp/examples/main/CMakeFiles/main.dir/depend:
	cd /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/llama.cpp/examples/main /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main /home/robert/Documents/gpt4all/gpt4all/gpt4all-backend/build/llama.cpp/examples/main/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : llama.cpp/examples/main/CMakeFiles/main.dir/depend

