cmake_minimum_required(VERSION 3.18)
project(cubify_server LANGUAGES CXX)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)  # downloaded files get extraction-time stamps
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Static MSVC runtime: the server then runs without any VC++ redistributable.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# ---- Eigen (header-only): installed package, -DEIGEN3_INCLUDE_DIR=..., or download
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Eigen include directory (optional)")
find_package(Eigen3 3.3 NO_MODULE QUIET)
if(EIGEN3_INCLUDE_DIR)
  add_library(cubify_eigen INTERFACE)
  target_include_directories(cubify_eigen INTERFACE "${EIGEN3_INCLUDE_DIR}")
  message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")
elseif(TARGET Eigen3::Eigen)
  add_library(cubify_eigen INTERFACE)
  target_link_libraries(cubify_eigen INTERFACE Eigen3::Eigen)
  message(STATUS "Eigen: found package ${Eigen3_VERSION}")
else()
  include(FetchContent)
  message(STATUS "Eigen: downloading 3.4.0")
  FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    # only the headers are needed: point at a directory without a
    # CMakeLists.txt so Eigen's own build (tests, docs) is skipped
    SOURCE_SUBDIR no-cmake-build)
  FetchContent_MakeAvailable(eigen)
  add_library(cubify_eigen INTERFACE)
  target_include_directories(cubify_eigen INTERFACE "${eigen_SOURCE_DIR}")
endif()

find_package(Threads REQUIRED)

add_executable(cubify_server
  server.cpp
  cubic_stylizer.cpp
  cubic_stylizer.h
  protocol.h
  thin_walls.cpp
  thin_walls.h
  thread_pool.h)
target_link_libraries(cubify_server PRIVATE cubify_eigen Threads::Threads)

if(MSVC)
  target_compile_options(cubify_server PRIVATE /O2 /W3 /bigobj /permissive-)
  target_compile_definitions(cubify_server PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  target_compile_options(cubify_server PRIVATE -O3 -Wall)
endif()

# Drop the executable straight into the add-on's bin/ folder, where the
# Blender side looks for it ($<1:> stops multi-config generators from
# appending a Release/ subfolder).
set_target_properties(cubify_server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_SOURCE_DIR}/../bin>")
