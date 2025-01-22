
cmake_minimum_required(VERSION 3.23)

set(CMAKE_BUILD_TYPE_INIT Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Establish project

project(dense2vdb LANGUAGES C CXX)

include(GNUInstallDirs)

option(USE_OPENMP "Enable OpenMP Support" OFF)
option(USE_MPI "Enable MPI Support" OFF)
option(EXPORT_VDB "Enable exporting of VDB" OFF)
option(EXPORT_HISTOGRAM "Enable exporting of histograms" OFF)
option(USE_LOG "Enable logging" OFF)

find_package(OpenVDB REQUIRED COMPONENTS openvdb nanovdb)

if(USE_LOG)
  add_definitions(-DUSE_LOG)
endif()

if(EXPORT_VDB)
  add_definitions(-DEXPORT_VDB)
endif()

if(EXPORT_HISTOGRAM)
  add_definitions(-DEXPORT_HISTOGRAM)
endif()

if(USE_OPENMP)
  add_definitions(-DUSE_OPENMP)
endif()

if(USE_OPENMP OR USE_LOG)
  find_package(OpenMP REQUIRED)
endif()

if(USE_MPI)
  find_package(MPI REQUIRED)
  add_definitions(-DUSE_MPI)
endif()

# Library

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE dense2nvdb.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenVDB::openvdb)
if(DENSE2VDB_NANOVDB_INTERFACE_LIBRARY)
  target_link_libraries(${PROJECT_NAME} PUBLIC ${DENSE2VDB_NANOVDB_INTERFACE_LIBRARY})
else()
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenVDB::nanovdb)
endif()

if(USE_OPENMP OR USE_LOG)
  target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_MPI)
  target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
  target_include_directories(${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})
endif()

# Compressor executable

add_executable(compress_${PROJECT_NAME})
target_sources(compress_${PROJECT_NAME} PRIVATE main.cpp)
target_link_libraries(compress_${PROJECT_NAME} PRIVATE ${PROJECT_NAME})

if(USE_OPENMP OR USE_LOG)
  target_link_libraries(compress_${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_MPI)
  target_include_directories(compress_${PROJECT_NAME} PRIVATE ${MPI_INCLUDE_PATH})
  target_link_libraries(compress_${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
endif()
