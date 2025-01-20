
cmake_minimum_required(VERSION 3.23)

set(CMAKE_BUILD_TYPE_INIT Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Establish project

project(dense2vdb LANGUAGES C CXX)

include(GNUInstallDirs)

find_package(OpenVDB REQUIRED COMPONENTS openvdb nanovdb)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Library

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE dense2nvdb.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenVDB::openvdb)
if(DENSE2VDB_NANOVDB_INTERFACE_LIBRARY)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${DENSE2VDB_NANOVDB_INTERFACE_LIBRARY} OpenMP::OpenMP_CXX MPI::MPI_CXX)
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenVDB::nanovdb OpenMP::OpenMP_CXX MPI::MPI_CXX)
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})

# Compressor executable

add_executable(compress_${PROJECT_NAME})
target_sources(compress_${PROJECT_NAME} PRIVATE main.cpp)
target_link_libraries(compress_${PROJECT_NAME} PRIVATE ${PROJECT_NAME} OpenMP::OpenMP_CXX MPI::MPI_CXX)

target_include_directories(compress_${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})
