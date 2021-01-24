# CMake version
cmake_minimum_required(VERSION 3.8)

# Declare project
project(BabaIsAgent)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# 3rd-party
add_subdirectory(Libraries/baba-is-auto)

# Include directories
include_directories(Includes)

# Project module
add_subdirectory(Sources/BabaIsAgent)
