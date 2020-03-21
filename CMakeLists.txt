cmake_minimum_required(VERSION 3.7)
project(GrafikaHazi1)

set(CMAKE_CXX_STANDARD 11)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Ha szeretnél több warningot
#set(CMAKE_CXX_FLAGS "-Wall -Wextra")

# optimizing options
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(SOURCE_FILES framework.h framework.cpp Skeleton.cpp)

include_directories(include)
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Linux)
	link_directories(lib/Linux)
endif()
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
	link_directories(lib/Windows)
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Linux)
	target_link_libraries(${PROJECT_NAME} glut GLEW GL) # dl pthread X11
endif()
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
	target_link_libraries(${PROJECT_NAME} opengl32 freeglut glew32)
endif()

