
cmake_minimum_required(VERSION 3.23)

set(CMAKE_BUILD_TYPE_INIT Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Establish project

project(dense2vdb LANGUAGES C CXX)

include(GNUInstallDirs)

find_package(OpenVDB REQUIRED)

# Add executable

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenVDB::openvdb)
