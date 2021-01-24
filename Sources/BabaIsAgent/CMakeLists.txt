set(target BabaIsAgent)

file(GLOB_RECURSE sources
	${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

add_executable(${target} ${sources})
