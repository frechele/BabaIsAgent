# CMake version
cmake_minimum_required(VERSION 3.8)

# Declare project
project(BabaIsAgent)

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Options
option(USE_TENSORRT "Build with TensorRT" OFF)

# Set output directories
set(DEFAULT_CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# 3rd-party
add_subdirectory(Libraries/baba-is-auto)
include_directories(Libraries/baba-is-auto/Includes)

include_directories(Libraries/json/single_include)
include_directories(Libraries/random/include)
include_directories(Libraries/Lyra/include)

if(USE_TENSORRT)
	include(CMake/FindTensorRT.cmake)

	if (TensorRT_FOUND)
		message("Using TensorRT")

		add_compile_definitions(USE_TENSORRT)
		find_package(CUDA REQUIRED)
	else()
		message("Cannot find TensorRT, disable TensorRT option")
	endif()
endif()

# Include directories
include_directories(Includes)

# Project module
add_subdirectory(Sources/BabaIsAgent)
